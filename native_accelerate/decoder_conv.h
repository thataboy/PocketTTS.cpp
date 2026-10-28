#pragma once

// Adapted from PocketTTS-RAVEN's AccelConv (Pantelis Kalogiros, 2026).
// MIT license and attribution: native_accelerate/RAVEN_LICENSE.
// This optional backend keeps convolution inputs, weights and outputs FP32.
#ifdef PTT_ACCELERATE
#include <Accelerate/Accelerate.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pocket_accel {

class ConvKernel {
public:
    explicit ConvKernel(bool elu) : elu_(elu) {}

    void Compute(OrtKernelContext* raw) {
        Ort::KernelContext ctx(raw);
        auto x = ctx.GetInput(0), w = ctx.GetInput(1), b = ctx.GetInput(2);
        const auto xs = x.GetTensorTypeAndShapeInfo().GetShape();
        const auto ws = w.GetTensorTypeAndShapeInfo().GetShape();
        const auto bs = b.GetTensorTypeAndShapeInfo().GetShape();
        if (xs.size() != 3 || ws.size() != 3 || bs.size() != 1 || xs[0] != 1 ||
            xs[1] != ws[2] || bs[0] != ws[1])
            throw std::runtime_error("AccelConv expects x [1,Cin,T], w [K,Cout,Cin], bias [Cout]");
        const int64_t cin = xs[1], tin = xs[2], ksize = ws[0], cout = ws[1];
        const int64_t tout = tin - ksize + 1;
        if (cin <= 0 || cin > INT_MAX || tin <= 0 || tin > INT_MAX ||
            ksize <= 0 || cout <= 0 || cout > INT_MAX || tout <= 0 ||
            cout > INT_MAX / tout)
            throw std::runtime_error("AccelConv dimensions exceed supported BLAS range");
        const int64_t shape[] = {1, cout, tout};
        auto output = ctx.GetOutput(0, shape, 3);
        float* out = output.GetTensorMutableData<float>();
        const float* input = x.GetTensorData<float>();
        const float* weight = w.GetTensorData<float>();
        const float* bias = b.GetTensorData<float>();
        for (int64_t c = 0; c < cout; ++c)
            std::fill(out + c * tout, out + (c + 1) * tout, bias[c]);
        for (int64_t k = 0; k < ksize; ++k)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        int(cout), int(tout), int(cin), 1.0f,
                        weight + k * cout * cin, int(cin), input + k, int(tin),
                        1.0f, out, int(tout));
        if (elu_) {
            // ELU(alpha=1), expressed with Accelerate vector functions.
            const int n = int(cout * tout);
            scratch_.resize(size_t(n));
            const float lo = -std::numeric_limits<float>::max();
            const float hi = std::numeric_limits<float>::max();
            const float zero = 0.0f, minus_one = -1.0f;
            vDSP_vclip(out, 1, &lo, &zero, scratch_.data(), 1, n);
            vvexpf(scratch_.data(), scratch_.data(), &n);
            vDSP_vsadd(scratch_.data(), 1, &minus_one, scratch_.data(), 1, n);
            vDSP_vclip(out, 1, &zero, &hi, out, 1, n);
            vDSP_vadd(out, 1, scratch_.data(), 1, out, 1, n);
        }
    }

private:
    bool elu_;
    std::vector<float> scratch_;
};

template<bool Elu>
struct ConvOp : Ort::CustomOpBase<ConvOp<Elu>, ConvKernel> {
    void* CreateKernel(const OrtApi&, const OrtKernelInfo*) const { return new ConvKernel(Elu); }
    const char* GetName() const { return Elu ? "AccelConvElu" : "AccelConv"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
    size_t GetInputTypeCount() const { return 3; }
    ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
};

} // namespace pocket_accel
#endif
