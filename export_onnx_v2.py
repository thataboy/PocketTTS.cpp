#!/usr/bin/env python3
"""Export pinned pocket3 LSD checkpoints as versioned FP32/INT8 C++ bundles.

Use --source ../pocket3. Export patches are scoped; validation uses the original
Python implementations with independent states. CPU graph optimizations and INT8 are enabled by default.
"""

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from onnx import helper as h
import torch
from torch import nn
from torch.nn import functional as F

PINNED_REVISION = "b94db177bc0f77517d9b124843e25605d6f9aed8"


def flatten(model, capacity, prefix=None):
    from pocket_tts.modules.stateful_module import init_states

    states = init_states(model, 1, capacity)
    if prefix:
        states = {n: s for n, s in states.items() if n.startswith(prefix)}
    layout = [(n, k) for n in sorted(states) for k in sorted(states[n])]
    tensors = [
        torch.nan_to_num(states[n][k])
        if states[n][k].is_floating_point()
        else states[n][k]
        for n, k in layout
    ]
    return layout, tensors


def unpack(layout, tensors):
    state = {}
    for (name, key), tensor in zip(layout, tensors):
        state.setdefault(name, {})[key] = tensor
    return state


@contextlib.contextmanager
def export_patches():
    """Functional convolutions and bounded attention caches for ONNX tracing."""
    from pocket_tts.modules.attention import StreamingMultiheadAttention
    from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d

    originals = []

    def patch(cls, name, method):
        originals.append((cls, name, getattr(cls, name)))
        setattr(cls, name, method)

    def conv(self, x, model_state):
        state = (
            self.init_state(x.shape[0], 0)
            if model_state is None
            else self.get_state(model_state)
        )
        prev = state["previous"]
        if prev.shape[-1]:
            if self.pad_mode == "replicate":
                prev = torch.where(
                    state["first"].view(-1, 1, 1), x[..., :1].expand_as(prev), prev
                )
            x = torch.cat([prev, x], -1)
        y = self.conv(x)
        if prev.shape[-1]:
            state["previous"] = x[..., -prev.shape[-1] :]
        if self.pad_mode == "replicate":
            state["first"] = torch.zeros_like(state["first"])
        return y

    def convtr(self, x, model_state):
        state = self.get_state(model_state)
        incoming = state["partial"]
        y = self.convtr(x)
        n = incoming.shape[-1]
        if n:
            state["partial"] = y[..., -n:] - (
                self.convtr.bias[:, None] if self.convtr.bias is not None else 0
            )
            y = torch.cat([y[..., :n] + incoming, y[..., n:-n]], -1)
        return y

    def attention(self, query, model_state, attn_mask=None):
        b, length, _ = query.shape
        q, k, v = (
            self.in_proj(query)
            .view(b, length, 3, self.num_heads, self.dim_per_head)
            .unbind(2)
        )
        state = None if model_state is None else self.get_state(model_state)
        offset = (
            query.new_zeros((), dtype=torch.int64)
            if state is None
            else state["offset"][0]
        )
        q, k = self.rope(q, k, offset=offset)
        if state is None:
            positions = torch.arange(length, device=query.device)
            keys, values = k, v
        else:
            split = "cache_k" in state
            cache_k, cache_v = (
                (state["cache_k"], state["cache_v"])
                if split
                else state["cache"].unbind(0)
            )
            cap = cache_k.shape[1]
            absolute = offset + torch.arange(length, device=query.device)
            write = absolute % cap if self.context is not None else absolute
            idx = write.view(1, length, 1, 1).expand_as(k)
            keys = cache_k.scatter(1, idx, k)
            values = cache_v.scatter(1, idx, v)
            if split:
                state["cache_k"], state["cache_v"] = keys, values
            else:
                state["cache"] = torch.stack([keys, values])
            if self.context is None:
                # Full causal attention only needs the populated prefix. Keeping
                # capacity-sized K/V here wastes work on every generated frame.
                valid = offset + length
                keys, values = keys[:, :valid], values[:, :valid]
                positions = torch.arange(valid, device=query.device)
            else:
                positions = torch.arange(cap, device=query.device)
            if self.context is not None:
                last = offset + length - 1
                positions = last - ((last - positions) % cap)
        queries = offset + torch.arange(length, device=query.device)
        delta = queries[:, None] - positions[None, :]
        mask = (positions[None, :] >= 0) & (delta >= 0)
        if self.context is not None:
            mask = mask & (delta < self.context)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2), mask
        )
        return self.out_proj(out.transpose(1, 2).reshape(b, length, self.embed_dim))

    patch(StreamingConv1d, "forward", conv)
    patch(StreamingConvTranspose1d, "forward", convtr)
    patch(StreamingMultiheadAttention, "forward", attention)
    try:
        yield
    finally:
        for cls, name, method in reversed(originals):
            setattr(cls, name, method)


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.mimi = model.mimi
        self.projection = model.flow_lm.speaker_proj_weight
        self.prefix = (
            model.flow_lm.bos_before_voice
            if model.flow_lm.insert_bos_before_voice
            else None
        )

    def forward(self, audio):
        # Dynamic padding: upstream pad_for_conv1d uses math.ceil on a Python
        # shape, which freezes the length in the legacy exporter.
        n = self.mimi.frame_size
        audio = F.pad(audio, (0, (n - audio.shape[-1] % n) % n))
        x = self.mimi.encoder(audio, None)
        (x,) = self.mimi.encoder_transformer(x, None)
        x = self.mimi._to_framerate(x).transpose(-1, -2)
        x = F.linear(x.float(), self.projection)
        return torch.cat([self.prefix, x], 1) if self.prefix is not None else x


class Main(nn.Module):
    def __init__(self, model, capacity):
        super().__init__()
        self.lm = model.flow_lm
        layout, initial = flatten(self.lm, capacity)
        self.layout, self.initial = [], []
        for (name, key), value in zip(layout, initial):
            if key == "cache":
                for index, split_key in enumerate(("cache_k", "cache_v")):
                    self.layout.append((name, split_key))
                    self.initial.append(value[index].contiguous())
            else:
                self.layout.append((name, key))
                self.initial.append(value)

    def forward(self, sequence, text_embeddings, *states):
        state = unpack(self.layout, states)
        # The independent Python reference keeps upstream's combined cache.
        # Export alone uses split buffers, so ScatterElements can write directly
        # into the native engine's persistent K/V allocations.
        if not torch.onnx.is_in_onnx_export():
            for module_state in state.values():
                if "cache_k" in module_state:
                    module_state["cache"] = torch.stack(
                        [module_state.pop("cache_k"), module_state.pop("cache_v")]
                    )
        sequence = torch.where(torch.isnan(sequence), self.lm.bos_emb, sequence)
        x = torch.cat([text_embeddings, self.lm.input_linear(sequence)], 1)
        x = self.lm.transformer(x, state)
        for module_state in state.values():
            if "offset" in module_state:
                module_state["offset"] = module_state["offset"] + x.shape[1]
            if "cache" in module_state:
                module_state["cache_k"], module_state["cache_v"] = module_state.pop(
                    "cache"
                ).unbind(0)
        c = self.lm.out_norm(x[:, -1])
        return (
            c.squeeze(0),
            self.lm.out_eos(c).squeeze(0),
            *(state[n][k] for n, k in self.layout),
        )


class Flow(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net = model.flow_lm.flow_net

    def forward(self, c, s, t, x):
        return self.net(c, s, t, x)


class Decoder(nn.Module):
    def __init__(self, model, capacity=512):
        super().__init__()
        self.mimi = model.mimi
        self.mean = model.flow_lm.emb_mean
        self.std = model.flow_lm.emb_std
        self.ratio = int(model.mimi.encoder_frame_rate / model.mimi.frame_rate)
        self.layout, self.initial = flatten(
            self.mimi, capacity, ("decoder.", "decoder_transformer.", "upsample.")
        )

    def forward(self, latent, *states):
        state = unpack(self.layout, states)
        audio = self.mimi.decode_from_latent(latent * self.std + self.mean, state)
        for name in state:
            if "self_attn" in name:
                state[name]["offset"] = (
                    state[name]["offset"] + latent.shape[1] * self.ratio
                )
        return (audio, *(state[n][k] for n, k in self.layout))


def export(wrapper, inputs, path, names, outputs, dynamic):
    print(f"Exporting {path.name}", flush=True)
    with torch.no_grad(), export_patches(), warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        torch.onnx.export(
            wrapper.eval(),
            inputs,
            str(path),
            input_names=names,
            output_names=outputs,
            dynamic_axes=dynamic,
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )


def session(path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts)


def check(label, reference, actual, report, atol=1e-4):
    a = np.asarray(reference)
    b = np.asarray(actual)
    if a.shape != b.shape:
        raise AssertionError(f"{label}: shapes {a.shape} != {b.shape}")
    err = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0
    report.append({"check": label, "max_abs_error": err})
    if not np.allclose(a, b, atol=atol, rtol=1e-4):
        raise AssertionError(f"{label}: max error {err}")


def validate(model, directory, main, dec):
    """Unpatched upstream methods are the reference. Never validate a patch against itself."""
    from pocket_tts.modules.stateful_module import increment_steps, init_states

    report = []
    encoder = session(directory / "mimi_encoder.onnx")
    rng = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for samples in (1920, 23041, 48017):
            audio = torch.randn(1, 1, samples, generator=rng) * 0.05
            expected = model._encode_audio(audio)
            if model.flow_lm.insert_bos_before_voice:
                expected = torch.cat([model.flow_lm.bos_before_voice, expected], 1)
            check(
                f"encoder_{samples}",
                expected.numpy(),
                encoder.run(None, {"audio": audio.numpy()})[0],
                report,
            )
        text = (
            "Bonjour, le monde !"
            if directory.name.startswith("french")
            else "Hello, world!"
        )
        ids = model.flow_lm.conditioner.prepare(text)
        embeddings = model.flow_lm.conditioner(ids)
        check(
            "text",
            embeddings.numpy(),
            session(directory / "text_conditioner.onnx").run(
                None, {"token_ids": ids.numpy()}
            )[0],
            report,
        )
        main_session = session(directory / "flow_lm_main.onnx")
        state = [x.clone() for x in main.initial]
        ort_state = [x.numpy().copy() for x in state]
        for step in range(12):
            seq = (
                torch.full((1, 1, model.flow_lm.ldim), float("nan"))
                if step == 2
                else torch.randn(1, 1, model.flow_lm.ldim, generator=rng)
            )
            emb = (
                expected
                if step == 0
                else embeddings
                if step == 1
                else torch.empty(1, 0, model.flow_lm.dim)
            )
            if step < 2:
                seq = torch.empty(1, 0, model.flow_lm.ldim)
            ref = main(seq, emb, *state)
            feed = {
                "sequence": seq.numpy(),
                "text_embeddings": emb.numpy(),
                **{f"state_{i}": x for i, x in enumerate(ort_state)},
            }
            out = main_session.run(None, feed)
            for i, (a, b) in enumerate(zip(ref, out)):
                check(f"main_{step}_{i}", a.numpy(), b, report, 3e-4)
            state = list(ref[2:])
            ort_state = out[2:]
        flow = session(directory / "flow_lm_flow.onnx")
        c, s, t, x = (
            torch.randn(1024, generator=rng),
            torch.zeros(1, 1),
            torch.ones(1, 1),
            torch.randn(1, 32, generator=rng),
        )
        check(
            "flow",
            model.flow_lm.flow_net(c, s, t, x).numpy(),
            flow.run(
                None, {"c": c.numpy(), "s": s.numpy(), "t": t.numpy(), "x": x.numpy()}
            )[0],
            report,
        )
        decoder = session(directory / "mimi_decoder.onnx")
        ref_state = init_states(model.mimi, 1, 2048)
        for state_dict in ref_state.values():
            for k, v in state_dict.items():
                if v.is_floating_point():
                    state_dict[k] = torch.nan_to_num(v)
        ort_state = [x.numpy().copy() for x in dec.initial]
        for step, frames in enumerate([1] * 40 + [2, 15, 16]):
            latent = torch.randn(1, frames, 32, generator=rng) * 0.1
            ref = model.mimi.decode_from_latent(
                latent * model.flow_lm.emb_std + model.flow_lm.emb_mean, ref_state
            )
            increment_steps(model.mimi, ref_state, frames * 16)
            out = decoder.run(
                None,
                {
                    "latent": latent.numpy(),
                    **{f"state_{i}": x for i, x in enumerate(ort_state)},
                },
            )
            check(f"decoder_{step}", ref.numpy(), out[0], report, 3e-4)
            ort_state = out[1:]
    (directory / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Passed {len(report)} independent component checks", flush=True)


def initial_states(ort_session):
    types = {
        "tensor(float)": np.float32,
        "tensor(int64)": np.int64,
        "tensor(bool)": np.bool_,
    }
    return {
        i.name: np.zeros(i.shape, dtype=types[i.type])
        for i in ort_session.get_inputs()
        if i.name.startswith("state_")
    }


def run_state(ort_session, inputs, state, outputs):
    result = ort_session.run(None, {**inputs, **state})
    count = len(outputs)
    return result[:count], {
        f"state_{i}": value for i, value in enumerate(result[count:])
    }


def validate_int8(directory, manifest):
    """Compare teacher-forced real-voice trajectories and codec streaming to validated FP32."""
    from pocket_tts.data.audio_utils import end_on_pause
    from scipy.io import wavfile
    from tokenizers import Tokenizer

    # Optional local prompt fixture; use deterministic low-level inputs when a
    # caller exports away from this workspace. End-to-end ASR is a separate gate.
    fixture = Path(".cache/milestone-validation") / (
        manifest["language"] + "-voice.wav"
    )
    rng = np.random.default_rng(42)
    if fixture.exists():
        rate, audio = wavfile.read(fixture)
        if rate != 24000 or audio.ndim != 1 or audio.dtype != np.float32:
            raise ValueError("INT8 prompt fixture must be mono float32 at 24 kHz")
        audio = end_on_pause(torch.from_numpy(audio)[None], rate).numpy()[None]
    else:
        audio = rng.normal(0, 0.05, (1, 1, 24000)).astype(np.float32)
    prompt = session(directory / "mimi_encoder.onnx").run(None, {"audio": audio})[0]
    text = (
        "Bonjour le monde. Ceci est un test de synthèse vocale en français."
        if manifest["language"].startswith("french")
        else "Hello, world. This is a test of the new speech model."
    )
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokens = np.array([tokenizer.encode(text).ids], dtype=np.int64)
    embeddings = session(directory / "text_conditioner.onnx").run(
        None, {"token_ids": tokens}
    )[0]
    mains = [
        session(directory / f"flow_lm_main{suffix}.onnx") for suffix in ("", "_int8")
    ]
    flows = [
        session(directory / f"flow_lm_flow{suffix}.onnx") for suffix in ("", "_int8")
    ]
    decoders = [
        session(directory / f"mimi_decoder{suffix}.onnx") for suffix in ("", "_int8")
    ]
    main_states = [initial_states(s) for s in mains]
    dec_states = [initial_states(s) for s in decoders]
    for embedding in (prompt, embeddings):
        inputs = {
            "sequence": np.empty((1, 0, 32), np.float32),
            "text_embeddings": embedding,
        }
        for i, s in enumerate(mains):
            _, main_states[i] = run_state(
                s, inputs, main_states[i], ["conditioning", "eos_logit"]
            )
    latent = np.full((1, 1, 32), np.nan, np.float32)
    conditioning, flow_values, audio_values, eos_values = (
        [[], []],
        [[], []],
        [[], []],
        [[], []],
    )
    for step in range(96):
        current = []
        for i, s in enumerate(mains):
            output, main_states[i] = run_state(
                s,
                {
                    "sequence": latent,
                    "text_embeddings": np.empty((1, 0, 1024), np.float32),
                },
                main_states[i],
                ["conditioning", "eos_logit"],
            )
            conditioning[i].append(output[0])
            eos_values[i].append(output[1])
            current.append(output[0])
        noise = np.zeros((1, 32), np.float32)
        values = []
        for i, s in enumerate(flows):
            value = s.run(
                None,
                {
                    "c": current[0],
                    "s": np.zeros((1, 1), np.float32),
                    "t": np.ones((1, 1), np.float32),
                    "x": noise,
                },
            )[0]
            flow_values[i].append(value)
            values.append(value)
        latent = values[0][None]
        for i, s in enumerate(decoders):
            out, dec_states[i] = run_state(
                s, {"latent": latent}, dec_states[i], ["audio_frame"]
            )
            audio_values[i].append(out[0].reshape(-1))

    def metrics(pair):
        a, b = (np.concatenate(v).astype(np.float64).reshape(-1) for v in pair)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError("Non-finite INT8 validation output")
        error = float(np.sqrt(np.mean((a - b) ** 2)))
        rms = float(np.sqrt(np.mean(a * a)))
        return {
            "rmse": error,
            "relative_rmse": error / max(rms, 1e-12),
            "snr_db": float(20 * np.log10(max(rms, 1e-12) / max(error, 1e-12))),
            "cosine": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)),
            "max_abs_error": float(np.max(np.abs(a - b))),
        }

    report = {
        "frames": 96,
        "prompt_fixture": str(fixture)
        if fixture.exists()
        else "deterministic synthetic audio",
        "conditioning": metrics(conditioning),
        "flow": metrics(flow_values),
        "decoder": metrics(audio_values),
        "eos": metrics(eos_values),
    }
    if (
        report["conditioning"]["cosine"] < 0.99
        or report["conditioning"]["relative_rmse"] > 0.15
    ):
        raise RuntimeError(
            f"INT8 backbone error exceeds limits: {report['conditioning']}"
        )
    if report["flow"]["relative_rmse"] > 0.15:
        raise RuntimeError(f"INT8 flow error exceeds limits: {report['flow']}")
    # The decoder candidate is measured for the report, but the supported INT8
    # variant retains the validated FP32 codec to preserve waveform quality.
    report["decoder_candidate_selected"] = False
    print(json.dumps(report, indent=2), flush=True)
    for reference, quantized in zip(
        (mains[0], flows[0], decoders[0]), (mains[1], flows[1], decoders[1])
    ):
        for method in ("get_inputs", "get_outputs"):
            expected = [(x.name, x.type, x.shape) for x in getattr(reference, method)()]
            actual = [(x.name, x.type, x.shape) for x in getattr(quantized, method)()]
            if expected != actual:
                raise RuntimeError(
                    f"Quantization changed the streaming interface: {method}"
                )
    (directory / "validation_int8.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def add_int8(directory, manifest):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.onnx_model import ONNXModel

    coverage = {}
    for component in ("flow_lm_main", "flow_lm_flow", "mimi_decoder"):
        source = directory / (component + ".onnx")
        destination = directory / (component + "_int8.onnx")
        print(f"Quantizing {component}: per-channel QInt8 MatMul", flush=True)
        model = ONNXModel(onnx.load(source))
        # torch exports rank-two Linear as Gemm. Without lowering those, most
        # of the flow MLP would silently remain FP32.
        model.replace_gemm_with_matmul()
        quantize_dynamic(
            model.model,
            str(destination),
            per_channel=True,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
            extra_options={"MatMulConstBOnly": True},
        )
        quantized = onnx.load(destination)
        onnx.checker.check_model(quantized)
        count = sum(n.op_type == "MatMulInteger" for n in quantized.graph.node)
        if count == 0:
            raise RuntimeError(f"No dense weights were quantized in {component}")
        coverage[component] = {
            "matmul_integer_nodes": count,
            "fp32_bytes": source.stat().st_size,
            "int8_bytes": destination.stat().st_size,
        }
        session(destination)  # catch unsupported kernels/interfaces before publication
    report = validate_int8(directory, manifest)
    components = (
        "mimi_encoder",
        "text_conditioner",
        "flow_lm_main",
        "flow_lm_flow",
        "mimi_decoder",
    )
    manifest["variants"] = {
        "fp32": {name: name + ".onnx" for name in components},
        "int8": {
            name: name
            + ("_int8" if name in ("flow_lm_main", "flow_lm_flow") else "")
            + ".onnx"
            for name in components
        },
    }
    # Quantizing the small codec attention block saves little and has a much
    # larger waveform error than quantizing the backbone. Retain it in FP32.
    (directory / "mimi_decoder_int8.onnx").unlink()
    for name, details in coverage.items():
        details["selected"] = name != "mimi_decoder"
    manifest["precision"] = ["fp32", "int8"]
    manifest["quantization"] = {
        "method": "dynamic",
        "weights": "QInt8",
        "activations": "QUInt8 dynamic for quantized MatMul",
        "per_channel": True,
        "kv_cache": "float32",
        "decoder": "float32",
        "coverage": coverage,
        "component_validation": report,
    }
    return manifest


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

def validate_decoder(source, target):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False
    sessions = [
        ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
        for p in (source, target)
    ]
    dtypes = {
        "tensor(float)": np.float32,
        "tensor(int64)": np.int64,
        "tensor(bool)": np.bool_,
    }
    states = [
        {i.name: np.zeros(i.shape, dtypes[i.type]) for i in s.get_inputs()[1:]}
        for s in sessions
    ]
    rng = np.random.default_rng(123)
    signal_energy = error_energy = max_error = 0.0
    batches = [1, 4, 4, 8, 15, 16, 2, 1, 4, 16, 4] * 4
    for frames in batches:
        latent = rng.normal(0, 1, (1, frames, 32)).astype(np.float32)
        outputs = []
        for index, session in enumerate(sessions):
            result = session.run(None, {"latent": latent, **states[index]})
            outputs.append(result)
            states[index] = dict(zip(states[index], result[1:]))
        for ref, actual in zip(*outputs):
            if ref.dtype == np.float32:
                np.testing.assert_allclose(actual, ref, atol=2e-5, rtol=2e-4)
            else:
                np.testing.assert_array_equal(actual, ref)
        ref, actual = outputs[0][0].astype(np.float64), outputs[1][0].astype(np.float64)
        error = actual - ref
        signal_energy += np.sum(ref**2)
        error_energy += np.sum(error**2)
        max_error = max(max_error, float(np.max(np.abs(error))))
    snr = float(10 * np.log10(signal_energy / max(error_energy, 1e-30)))
    if snr < 95 or max_error > 2e-5:
        raise ValueError(
            f"Decoder numerical acceptance failed: SNR={snr}, max_error={max_error}"
        )
    return {
        "snr_db": snr,
        "max_absolute_error": max_error,
        "latent_frames": sum(batches),
        "batches": batches,
    }

def rewrite_main(src, dst):
    m = onnx.load(src)
    if any(i.type.tensor_type.shape.dim[0].dim_value != 1 for i in m.graph.input[:2]):
        raise ValueError("Expected batch-one main model")
    nodes = {n.name: n for n in m.graph.node}
    new = []
    count = 0
    axis = "cpu_layout/axis"
    m.graph.initializer.append(
        onnx.numpy_helper.from_array(np.array([0], dtype="int64"), axis)
    )
    for n in m.graph.node:
        if n.name.endswith("/self_attn/MatMul") or n.name.endswith(
            "/self_attn/MatMul_1"
        ):
            first = n.name.endswith("/MatMul")
            p = n.name.rsplit("/", 1)[0] + "/"
            if first:
                mul = nodes[p + "Mul_1"]
                tr = nodes[p + "Transpose_2"]
                if mul.input[0] != tr.output[0] or list(
                    next(a.ints for a in tr.attribute if a.name == "perm")
                ) != [0, 2, 3, 1]:
                    raise ValueError("Unexpected key transpose/scaling")
                mul.input[0] = tr.input[0]
                rhs = n.input[1]
            else:
                tr = nodes[p + "Transpose_1"]
                if n.input[1] != tr.output[0] or list(
                    next(a.ints for a in tr.attribute if a.name == "perm")
                ) != [0, 2, 1, 3]:
                    raise ValueError("Unexpected value transpose")
                rhs = tr.input[0]
            pre = n.name + "/strided/"
            new.extend(
                [
                    h.make_node("Squeeze", [n.input[0], axis], [pre + "a"]),
                    h.make_node("Squeeze", [rhs, axis], [pre + "b"]),
                    h.make_node(
                        "FusedMatMul",
                        [pre + "a", pre + "b"],
                        [pre + "out"],
                        name=pre + "MatMul",
                        domain="com.microsoft",
                        transBatchB=1,
                        transB=int(first),
                    ),
                    h.make_node("Unsqueeze", [pre + "out", axis], list(n.output)),
                ]
            )
            count += 1
        else:
            new.append(n)
    needed = {o.name for o in m.graph.output}
    kept = []
    for n in reversed(new):
        if any(o in needed for o in n.output):
            kept.append(n)
            needed.update(n.input)
    del m.graph.node[:]
    m.graph.node.extend(reversed(kept))
    del m.graph.value_info[:]
    if not any(o.domain == "com.microsoft" for o in m.opset_import):
        m.opset_import.append(h.make_opsetid("com.microsoft", 1))
    if count != 12:
        raise ValueError(f"Expected 12 attention products, found {count}")
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    return count

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
                np.int64 if i.type == "tensor(int64)" else np.float32,
            )
            for i in state_inputs
        }

    states = [initial(), initial()]
    rng = np.random.default_rng(123)
    cases = [(0, 48), (0, 24)] + [(1, 0)] * 48 + [(4, 0), (0, 17), (1, 0)]
    checked = 0
    for step, (frames, tokens) in enumerate(cases):
        sequence = rng.normal(0, 0.1, (1, frames, 32)).astype(np.float32)
        if step == 2:
            sequence.fill(np.nan)
        text = rng.normal(0, 0.1, (1, tokens, 1024)).astype(np.float32)
        outputs = [
            s.run(None, {"sequence": sequence, "text_embeddings": text, **state})
            for s, state in zip(sessions, states)
        ]
        for a, b in zip(*outputs):
            np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
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
        for a, b in zip(*outputs):
            np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
        checked += 1
    return {
        "bit_exact": True,
        "calls": checked,
        "capacity": capacity,
        "ort_version": ort.__version__,
    }

def optimize_cpu(directory, manifest):
    """Apply and validate the retained CPU rewrites before publishing a bundle."""
    with tempfile.TemporaryDirectory(prefix="cpu-rewrite-") as temp:
        original = Path(temp)
        if not manifest.get("decoder_optimization"):
            name = "mimi_decoder.onnx"
            shutil.copy2(directory / name, original / name)
            nodes = rewrite_decoder(original / name, directory / name)
            report = validate_decoder(original / name, directory / name)
            report["nodes"] = nodes
            (directory / "decoder_cpu_validation.json").write_text(json.dumps(report, indent=2) + "\n")
            manifest["decoder_optimization"] = "phase_conv_v1"
        if not manifest.get("main_optimization"):
            names = {"flow_lm_main.onnx"}
            names.update(v["flow_lm_main"] for v in manifest.get("variants", {}).values())
            reports = {}
            for name in sorted(names):
                shutil.copy2(directory / name, original / name)
                count = rewrite_main(original / name, directory / name)
                reports[name] = validate_main(original / name, directory / name)
                reports[name]["attention_products"] = count
            (directory / "main_cpu_validation.json").write_text(json.dumps(reports, indent=2) + "\n")
            manifest["main_optimization"] = "strided_attention_v1"
    return manifest


def optimize_bundle(source, target):
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("schema_version") != 2 or manifest.get("main_cache_layout") != "split_fp32_v1":
        raise ValueError("Expected a v2 split-cache bundle")
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=target.name + "-", dir=target.parent) as temp:
        directory = Path(temp) / "bundle"
        shutil.copytree(source, directory)
        optimize_cpu(directory, manifest)
        write_manifest(directory, manifest)
        directory.rename(target)
    print(f"Validated CPU bundle: {target.resolve()}", flush=True)


def write_manifest(directory, manifest):
    manifest = {k: v for k, v in manifest.items() if k not in ("files", "fingerprint")}
    # Sizes are descriptive metadata; the runtime does not hash model files.
    manifest["files"] = {
        p.name: p.stat().st_size for p in sorted(directory.iterdir())
        if p.is_file() and p.name != "manifest.json"
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def quantize_bundle(source, target, python_source):
    """Build a new validated bundle without modifying the source bundle."""
    manifest = json.loads((source / "manifest.json").read_text())
    if (
        manifest["upstream_revision"] != PINNED_REVISION
        or manifest["flow_type"] != "lsd"
    ):
        raise ValueError("Unsupported source bundle")
    revision = subprocess.check_output(
        ["git", "-C", str(python_source.resolve()), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != manifest["upstream_revision"]:
        raise ValueError("Python validation source revision differs from source bundle")
    sys.path.insert(0, str(python_source.resolve()))
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=target.name + "-", dir=target.parent
    ) as temp:
        directory = Path(temp)
        for name in manifest["files"]:
            if "_int8" not in name:
                shutil.copy2(source / name, directory / name)
        manifest = add_int8(directory, manifest)
        optimize_cpu(directory, manifest)
        write_manifest(directory, manifest)
        directory.rename(target)
    print(f"Validated FP32/INT8 bundle: {target.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../pocket3"))
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--language", default=None)
    choice.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-seq", type=int, default=1000)
    parser.add_argument(
        "--int8",
        action="store_true", default=True,
        help="Export FP32 and validated INT8 weights (default)",
    )
    parser.add_argument(
        "--quantize-from",
        type=Path,
        help="Quantize an existing validated FP32 bundle without loading checkpoint weights",
    )
    parser.add_argument("--fp32-only", action="store_false", dest="int8", help="Export only FP32 weights")
    parser.add_argument("--optimize-from", type=Path, help="Optimize an existing split-cache v2 bundle without re-exporting")
    args = parser.parse_args()
    if args.optimize_from:
        if args.quantize_from or args.language or args.config:
            parser.error("--optimize-from cannot be combined with --quantize-from/--language/--config")
        target = args.output_dir or Path("models-v2") / args.optimize_from.name
        optimize_bundle(args.optimize_from, target)
        return
    source = args.source.resolve()
    if args.quantize_from:
        if args.language or args.config:
            parser.error("--quantize-from is incompatible with --language/--config")
        target = args.output_dir or Path("models-v2-int8") / args.quantize_from.name
        quantize_bundle(args.quantize_from, target, source)
        return
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PINNED_REVISION:
        parser.error(f"expected pocket3 revision {PINNED_REVISION}, got {revision}")
    sys.path.insert(0, str(source))
    from pocket_tts import TTSModel
    from pocket_tts.utils.utils import download_if_necessary

    language = args.language or (None if args.config else "english")
    target = args.output_dir or Path("models-v2") / (language or args.config.stem)
    if target.exists():
        parser.error(f"output already exists: {target}; choose a new directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    model = TTSModel.load_model(
        language=language, config=args.config, quantize=False
    ).eval()
    if not model.has_voice_cloning:
        raise RuntimeError(
            "Voice-cloning checkpoint unavailable; refusing to export an invalid encoder"
        )
    config = model.config.model_dump()
    if config["flow_lm"]["flow"]["type"] != "lsd":
        raise ValueError("This exporter currently supports LSD checkpoints only")
    if (
        model.flow_lm.dim != 1024
        or model.flow_lm.ldim != 32
        or model.sample_rate != 24000
    ):
        raise ValueError("Milestone runtime supports 1024/32 dimensions at 24 kHz")
    with tempfile.TemporaryDirectory(
        prefix=target.name + "-", dir=target.parent
    ) as temp:
        directory = Path(temp)
        encoder = Encoder(model)
        backbone = Main(model, args.max_seq)
        decoder = Decoder(model)
        ids = model.flow_lm.conditioner.prepare("Hello.")
        export(
            encoder,
            (torch.randn(1, 1, 23040),),
            directory / "mimi_encoder.onnx",
            ["audio"],
            ["conditioning"],
            {"audio": {2: "samples"}, "conditioning": {1: "frames"}},
        )
        export(
            model.flow_lm.conditioner,
            (ids,),
            directory / "text_conditioner.onnx",
            ["token_ids"],
            ["embeddings"],
            {"token_ids": {1: "tokens"}, "embeddings": {1: "tokens"}},
        )
        for wrapper, name, args_, inputs, outputs, dynamic in [
            (
                backbone,
                "flow_lm_main",
                (torch.zeros(1, 1, 32), torch.zeros(1, 0, 1024), *backbone.initial),
                ["sequence", "text_embeddings"],
                ["conditioning", "eos_logit"],
                {"sequence": {1: "frames"}, "text_embeddings": {1: "tokens"}},
            ),
            (
                decoder,
                "mimi_decoder",
                (torch.zeros(1, 1, 32), *decoder.initial),
                ["latent"],
                ["audio_frame"],
                {"latent": {1: "frames"}, "audio_frame": {2: "samples"}},
            ),
        ]:
            inputs += [f"state_{i}" for i in range(len(wrapper.initial))]
            outputs += [f"out_state_{i}" for i in range(len(wrapper.initial))]
            export(
                wrapper, args_, directory / (name + ".onnx"), inputs, outputs, dynamic
            )
        export(
            Flow(model),
            (
                torch.randn(1024),
                torch.zeros(1, 1),
                torch.ones(1, 1),
                torch.randn(1, 32),
            ),
            directory / "flow_lm_flow.onnx",
            ["c", "s", "t", "x"],
            ["flow_dir"],
            {},
        )
        tokenizer = Path(
            download_if_necessary(config["flow_lm"]["lookup_table"]["tokenizer_path"])
        )
        if tokenizer.suffix != ".json":
            raise ValueError("Milestone exporter expects a JSON tokenizer")
        shutil.copyfile(tokenizer, directory / "tokenizer.json")
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        validate(model, directory, backbone, decoder)
        manifest = {
            "schema_version": 2,
            "language": language or args.config.stem,
            "upstream_revision": revision,
            "flow_type": "lsd",
            "precision": ["fp32"],
            "voice_prefix_in_encoder": model.flow_lm.insert_bos_before_voice,
            "main_cache_layout": "split_fp32_v1",
            "max_seq": args.max_seq,
            "sample_rate": 24000,
            "latent_dim": 32,
            "d_model": 1024,
            "tokenizer": "tokenizer.json",
            "config": config,
            "audio_preprocessing": "end_on_pause_v1",
        }
        if args.int8:
            manifest = add_int8(directory, manifest)
        optimize_cpu(directory, manifest)
        write_manifest(directory, manifest)
        directory.rename(target)
    print(f"Validated bundle: {target.resolve()}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
