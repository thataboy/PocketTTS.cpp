#!/usr/bin/env python3
"""Apply validated v2 CPU graph rewrites to existing legacy English ONNX models.

Preserves v1 weights, SentencePiece tokenization, split FP16 KV caches, model
filenames and interfaces. Writes a separate model directory; no checkpoint,
PyTorch, Rust, or multilingual runtime is required.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


def rewrite_decoder(source, target):
    model = onnx.load(source)
    weights = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    nodes, rewritten = [], []
    for node in model.graph.node:
        if node.op_type != "ConvTranspose":
            nodes.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        stride = attrs.get("strides", [1])[0]
        groups = attrs.get("group", 1)
        weight = weights[node.input[1]]
        if (
            weight.ndim != 3
            or weight.shape[2] != 2 * stride
            or attrs.get("pads", [0, 0]) != [0, 0]
            or attrs.get("dilations", [1]) != [1]
            or attrs.get("output_padding", [0]) != [0]
            or "output_shape" in attrs
            or attrs.get("auto_pad", b"NOTSET") != b"NOTSET"
        ):
            raise ValueError(f"Unsupported ConvTranspose geometry: {node.name}")
        cin, cout, _ = weight.shape
        # y[t*s+r] = x[t-1]*W[s+r] + x[t]*W[r]. Conv uses
        # correlation order, hence reversal of the two kernel halves.
        converted = (
            weight.reshape(groups, cin // groups, cout, 2, stride)
            .transpose(0, 2, 4, 1, 3)[..., ::-1]
            .copy()
            .reshape(cout * groups * stride, cin // groups, 2)
        )
        prefix = node.name + "/cpu_phase"
        weight_name = prefix + "/weight"
        model.graph.initializer.append(numpy_helper.from_array(converted, weight_name))
        inputs = [node.input[0], weight_name]
        if len(node.input) == 3:
            bias_name = prefix + "/bias"
            model.graph.initializer.append(
                numpy_helper.from_array(
                    np.repeat(weights[node.input[2]], stride), bias_name
                )
            )
            inputs.append(bias_name)
        for name, shape in (
            ("split", [0, cout * groups, stride, -1]),
            ("merge", [0, cout * groups, -1]),
        ):
            model.graph.initializer.append(
                numpy_helper.from_array(np.array(shape, np.int64), prefix + "/" + name)
            )
        nodes.extend(
            [
                helper.make_node(
                    "Conv",
                    inputs,
                    [prefix + "/conv"],
                    name=prefix + "/Conv",
                    group=groups,
                    kernel_shape=[2],
                    strides=[1],
                    pads=[1, 1],
                ),
                helper.make_node(
                    "Reshape",
                    [prefix + "/conv", prefix + "/split"],
                    [prefix + "/phases"],
                ),
                helper.make_node(
                    "Transpose",
                    [prefix + "/phases"],
                    [prefix + "/order"],
                    perm=[0, 1, 3, 2],
                ),
                helper.make_node(
                    "Reshape", [prefix + "/order", prefix + "/merge"], list(node.output)
                ),
            ]
        )
        rewritten.append(node.name)
    if not rewritten:
        raise ValueError(
            "No ConvTranspose nodes found; bundle may already be optimized"
        )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {x for n in nodes for x in n.input} | {x.name for x in model.graph.output}
    retained = [i for i in model.graph.initializer if i.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    onnx.checker.check_model(model)
    onnx.save(model, target)
    return rewritten

class Comparison:
    """Accumulate floating error over complete streaming trajectories."""
    def __init__(self, names):
        self.stats = {name: {"signal": 0.0, "error": 0.0, "max_abs": 0.0} for name in names}
        self.bit_exact = True

    def add(self, names, reference, actual):
        for name, ref, value in zip(names, reference, actual):
            if ref.shape != value.shape or ref.dtype != value.dtype:
                raise ValueError("Rewrite changed an output interface")
            self.bit_exact &= np.array_equal(ref.view(np.uint8), value.view(np.uint8))
            if not np.issubdtype(ref.dtype, np.floating):
                np.testing.assert_array_equal(value, ref)
                continue
            if not np.isfinite(value).all() or not np.isfinite(ref).all():
                raise ValueError("Nonfinite model output")
            x, y = ref.astype(np.float64), value.astype(np.float64)
            err = y - x
            r = self.stats[name]
            r["signal"] += float(np.sum(x * x))
            r["error"] += float(np.sum(err * err))
            r["max_abs"] = max(r["max_abs"], float(np.max(np.abs(err))) if err.size else 0)

    def finish(self, relative_limit, absolute_limit):
        report = {}
        for name, r in self.stats.items():
            relative = float(np.sqrt(r["error"] / max(r["signal"], 1e-30)))
            if relative > relative_limit and r["max_abs"] > absolute_limit:
                raise ValueError(f"{name}: relative RMS error {relative} exceeds {relative_limit}")
            report[name] = {"relative_rmse": relative, "max_abs": r["max_abs"],
                            "snr_db": float(10 * np.log10(max(r["signal"], 1e-30) / max(r["error"], 1e-30)))}
        return report


def validate_decoder(source, target):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False
    sessions = [ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
                for p in (source, target)]
    for method in ["get_inputs", "get_outputs"]:
        interfaces = [[(i.name, i.type, i.shape) for i in getattr(s, method)()] for s in sessions]
        if interfaces[0] != interfaces[1]:
            raise ValueError("Rewrite changed decoder interface")
    dtype = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
             "tensor(int64)": np.int64, "tensor(bool)": np.bool_}
    states = [{i.name: np.zeros(i.shape, dtype[i.type]) for i in s.get_inputs()[1:]} for s in sessions]
    names = [i.name for i in sessions[0].get_outputs()]
    comparison = Comparison(names)
    rng = np.random.default_rng(123)
    batches = [1, 4, 4, 8, 15, 16, 2, 1, 4, 16, 4] * 4
    for frames in batches:
        latent = rng.normal(0, 1, (1, frames, 32)).astype(np.float32)
        outputs = [session.run(None, {"latent": latent, **state}) for session, state in zip(sessions, states)]
        comparison.add(names, *outputs)
        states = [dict(zip(state, result[1:])) for state, result in zip(states, outputs)]
    int8 = source.stem.endswith("_int8")
    # FP16 state rounding is not bit-exact. Dynamic INT8 scales amplify this;
    # bound numerical drift here and check real speech separately.
    metrics = comparison.finish(0.05 if int8 else 0.001, 0.001 if int8 else 0.0001)
    audio = metrics[names[0]]
    if audio["snr_db"] < (30 if int8 else 80):
        raise ValueError("Decoder audio SNR below acceptance threshold")
    return {"bit_exact": bool(comparison.bit_exact), "outputs": metrics,
            "latent_frames": sum(batches), "batches": batches,
            "audio_snr_db": audio["snr_db"], "audio_max_abs": audio["max_abs"]}


def validate_main(source, target):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sessions = [
        ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
        for p in (source, target)
    ]
    for method in ["get_inputs", "get_outputs"]:
        interfaces = [
            [(i.name, i.type, i.shape) for i in getattr(s, method)()] for s in sessions
        ]
        if interfaces[0] != interfaces[1]:
            raise ValueError("Rewrite changed the streaming interface")
    state_inputs = sessions[0].get_inputs()[2:]
    capacity = state_inputs[0].shape[1]

    def initial(offset=0):
        return {
            i.name: np.full(
                i.shape,
                offset if i.type == "tensor(int64)" else 0,
                {"tensor(int64)": np.int64, "tensor(float16)": np.float16, "tensor(float)": np.float32}[i.type],
            )
            for i in state_inputs
        }

    states = [initial(), initial()]
    rng = np.random.default_rng(123)
    cases = [(0, 48), (0, 24)] + [(1, 0)] * 48 + [(4, 0), (0, 17), (1, 0)]
    checked = 0
    names = [i.name for i in sessions[0].get_outputs()]
    comparison = Comparison(names)
    for step, (frames, tokens) in enumerate(cases):
        sequence = rng.normal(0, 0.1, (1, frames, 32)).astype(np.float32)
        if step == 2:
            sequence.fill(np.nan)
        text = rng.normal(0, 0.1, (1, tokens, 1024)).astype(np.float32)
        outputs = [
            s.run(None, {"sequence": sequence, "text_embeddings": text, **state})
            for s, state in zip(sessions, states)
        ]
        comparison.add(names, *outputs)
        states = [dict(zip((i.name for i in state_inputs), o[2:])) for o in outputs]
        checked += 1
    # Exercise both ends of the cache, including the final legal write.
    for offset, frames, tokens in [
        (1, 1, 0),
        (capacity // 2, 1, 0),
        (capacity - 1, 1, 0),
        (capacity - 8, 0, 8),
    ]:
        sequence = rng.normal(0, 0.1, (1, frames, 32)).astype(np.float32)
        text = rng.normal(0, 0.1, (1, tokens, 1024)).astype(np.float32)
        outputs = [
            s.run(
                None, {"sequence": sequence, "text_embeddings": text, **initial(offset)}
            )
            for s in sessions
        ]
        comparison.add(names, *outputs)
        checked += 1
    return {
        "bit_exact": bool(comparison.bit_exact),
        "outputs": comparison.finish(0.0001, 0.00001),
        "calls": checked,
        "capacity": capacity,
        "ort_version": ort.__version__,
    }


def rewrite_main(source, target):
    """Read sequence-major K/V directly, preserving both score scale factors."""
    model = onnx.load(source)
    producers = {output: node for node in model.graph.node for output in node.output}
    axis = "v1_cpu/batch_axis"
    model.graph.initializer.append(numpy_helper.from_array(np.array([0], np.int64), axis))
    nodes = []
    count = 0
    for node in model.graph.node:
        transpose = None
        score = False
        if node.op_type == "MatMul":
            rhs = producers.get(node.input[1])
            # Attention scores use scaled transposed K; values use transposed V.
            if rhs is not None and rhs.op_type == "Mul":
                candidate = producers.get(rhs.input[0])
                if candidate is not None and candidate.op_type == "Transpose":
                    perm = helper.get_attribute_value(next(a for a in candidate.attribute if a.name == "perm"))
                    if list(perm) == [0, 2, 3, 1]:
                        transpose, score = candidate, True
                        rhs.input[0] = transpose.input[0]
            elif rhs is not None and rhs.op_type == "Transpose":
                perm = helper.get_attribute_value(next(a for a in rhs.attribute if a.name == "perm"))
                if list(perm) == [0, 2, 1, 3]:
                    transpose = rhs
        if transpose is None:
            nodes.append(node)
            continue
        prefix = "v1_cpu/attention_" + str(count)
        right = node.input[1] if score else transpose.input[0]
        nodes.extend([
            helper.make_node("Squeeze", [node.input[0], axis], [prefix + "/a"]),
            helper.make_node("Squeeze", [right, axis], [prefix + "/b"]),
            helper.make_node("FusedMatMul", [prefix + "/a", prefix + "/b"], [prefix + "/out"],
                             name=prefix, domain="com.microsoft", transBatchB=1, transB=int(score)),
            helper.make_node("Unsqueeze", [prefix + "/out", axis], list(node.output)),
        ])
        count += 1
    if count != 12:
        raise ValueError(f"Expected twelve v1 attention products; found {count}")
    needed = {i.name for i in model.graph.output}
    kept = []
    for node in reversed(nodes):
        if any(output in needed for output in node.output):
            kept.append(node)
            needed.update(node.input)
    del model.graph.node[:]
    model.graph.node.extend(reversed(kept))
    del model.graph.value_info[:]
    if not any(i.domain == "com.microsoft" for i in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(model)
    onnx.save(model, target)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["decoder", "attention", "all"], default="all")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new model directory")
    if args.output.resolve().is_relative_to(args.models.resolve()):
        parser.error("Output must be outside the source directory")
    if not (args.models / "tokenizer.model").is_file():
        parser.error("Expected legacy models with tokenizer.model")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = args.models / "cpu_optimization.json"
    report = json.loads(previous.read_text()) if previous.exists() else {"components": {}, "stages": []}
    report["ort_version"] = ort.__version__
    report["stages"].append(args.stage)
    with tempfile.TemporaryDirectory(prefix="v1-cpu-", dir=args.output.parent) as temp:
        directory = Path(temp) / "models"
        directory.mkdir()
        for name in ["tokenizer.model", "mimi_encoder.onnx", "text_conditioner.onnx",
                     "flow_lm_flow.onnx", "flow_lm_flow_int8.onnx", "flow_lm_main.onnx",
                     "flow_lm_main_int8.onnx", "mimi_decoder.onnx", "mimi_decoder_int8.onnx"]:
            if (args.models / name).is_file():
                shutil.copy2(args.models / name, directory / name)
        groups = []
        if args.stage in ["decoder", "all"]:
            groups.append(("mimi_decoder", rewrite_decoder, validate_decoder))
        if args.stage in ["attention", "all"]:
            groups.append(("flow_lm_main", rewrite_main, validate_main))
        for component, rewrite, validate in groups:
            for suffix in ["", "_int8"]:
                name = component + suffix + ".onnx"
                if not (args.models / name).is_file():
                    continue
                if name in report["components"]:
                    continue
                print("Rewriting and validating", name, flush=True)
                rewritten = rewrite(args.models / name, directory / name)
                result = validate(args.models / name, directory / name)
                result["rewritten"] = rewritten
                report["components"][name] = result
        (directory / "cpu_optimization.json").write_text(json.dumps(report, indent=2) + "\n")
        directory.rename(args.output)
    print(json.dumps(report, indent=2))
    print("Validated models:", args.output.resolve())


if __name__ == "__main__":
    main()
