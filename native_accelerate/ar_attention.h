#pragma once

// Fused score/softmax/value tail for our split-cache AR graph. The ARM
// single-query kernel follows PocketTTS-RAVEN's attention implementation;
// attribution and MIT license: native_accelerate/RAVEN_LICENSE.
#ifdef PTT_ACCELERATE
#include <arm_neon.h>
#include <cmath>
#include "decoder_conv.h"

namespace pocket_accel {

class AttentionKernel {
public:
    void Compute(OrtKernelContext* raw) {
        Ort::KernelContext ctx(raw);
        auto qv = ctx.GetInput(0), kv = ctx.GetInput(1), vv = ctx.GetInput(2), mv = ctx.GetInput(3);
        auto sv = ctx.GetInput(4);
        const auto qs = qv.GetTensorTypeAndShapeInfo().GetShape();
        const auto ks = kv.GetTensorTypeAndShapeInfo().GetShape();
        const auto vs = vv.GetTensorTypeAndShapeInfo().GetShape();
        const auto ms = mv.GetTensorTypeAndShapeInfo().GetShape();
        if (qs.size() != 4 || ks.size() != 4 || qs[0] != 1 || ks[0] != 1 ||
            qs[3] != 64 || ks[3] != 64 || qs[1] != ks[2] || ks != vs ||
            ms.size() != 2 || ms[0] != qs[2] || ms[1] > ks[1] || ms[1] < qs[2] ||
            qs[1] <= 0 || qs[1] > 16 || qs[2] < 0 || qs[2] > 1000 || ks[1] > 1000 ||
            sv.GetTensorTypeAndShapeInfo().GetElementCount() != 1)
            throw std::runtime_error("SplitAttention expects q [1,H,Q,64], K/V [1,C,H,64], mask [Q,T], scalar key scale");
        const int heads = int(qs[1]), query = int(qs[2]), time = int(ms[1]), stride = heads*64;
        const int64_t shape[] = {1, query, stride};
        auto output = ctx.GetOutput(0, shape, 3);
        float* out = output.GetTensorMutableData<float>();
        if (query == 0) return;
        const float* q = qv.GetTensorData<float>();
        const float* k = kv.GetTensorData<float>();
        const float* v = vv.GetTensorData<float>();
        const float* mask = mv.GetTensorData<float>();
        const float key_scale = sv.GetTensorData<float>()[0];
        if (query > 1) {
            // Prefill reuses scaled keys for all queries. Single-token steps
            // scale each SIMD load directly, avoiding a full prefix copy.
            scaled_keys_.resize(size_t(time)*stride);
            vDSP_vsmul(k, 1, &key_scale, scaled_keys_.data(), 1, scaled_keys_.size());
        }
        scores_.resize(size_t(query)*time);
        for (int h = 0; h < heads; ++h) {
            const float* qh = q + h*query*64;
            const float* kh = k + h*64;
            const float* vh = v + h*64;
            float* oh = out + h*64;
            if (query == 1) {
                for (int t = 0; t < time; ++t) scores_[t] = dot64(qh, kh + t*stride, key_scale);
            } else {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            query, time, 64, 1.0f, qh, 64, scaled_keys_.data() + h*64, stride,
                            0.0f, scores_.data(), time);
            }
            for (int i = 0; i < query; ++i) softmax(scores_.data() + i*time, mask + i*time, time);
            if (query == 1) {
                float32x4_t acc[16];
                for (auto& a : acc) a = vdupq_n_f32(0.0f);
                for (int t = 0; t < time; ++t) {
                    const auto weight = vdupq_n_f32(scores_[t]);
                    const float* value = vh + t*stride;
                    for (int d = 0; d < 16; ++d)
                        acc[d] = vfmaq_f32(acc[d], weight, vld1q_f32(value + d*4));
                }
                for (int d = 0; d < 16; ++d) vst1q_f32(oh + d*4, acc[d]);
            } else {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            query, 64, time, 1.0f, scores_.data(), time, vh, stride,
                            0.0f, oh, stride);
            }
        }
    }

private:
    static float dot64(const float* a, const float* b, float scale) {
        float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
        for (int d = 0; d < 64; d += 16) {
            s0 = vfmaq_f32(s0, vld1q_f32(a+d), vmulq_n_f32(vld1q_f32(b+d),scale));
            s1 = vfmaq_f32(s1, vld1q_f32(a+d+4), vmulq_n_f32(vld1q_f32(b+d+4),scale));
            s2 = vfmaq_f32(s2, vld1q_f32(a+d+8), vmulq_n_f32(vld1q_f32(b+d+8),scale));
            s3 = vfmaq_f32(s3, vld1q_f32(a+d+12), vmulq_n_f32(vld1q_f32(b+d+12),scale));
        }
        return vaddvq_f32(vaddq_f32(vaddq_f32(s0,s1),vaddq_f32(s2,s3)));
    }

    static void softmax(float* row, const float* mask, int n) {
        vDSP_vadd(row, 1, mask, 1, row, 1, n);
        float max;
        vDSP_maxv(row, 1, &max, n);
        // Matches the exported IsNaN/Where guard on fully masked rows.
        if (!std::isfinite(max)) { std::fill(row, row+n, 0.0f); return; }
        max = -max;
        vDSP_vsadd(row, 1, &max, row, 1, n);
        vvexpf(row, row, &n);
        float sum;
        vDSP_sve(row, 1, &sum, n);
        const float scale = 1.0f/sum;
        vDSP_vsmul(row, 1, &scale, row, 1, n);
    }

    std::vector<float> scores_;
    std::vector<float> scaled_keys_;
};

struct AttentionOp : Ort::CustomOpBase<AttentionOp, AttentionKernel> {
    void* CreateKernel(const OrtApi&, const OrtKernelInfo*) const { return new AttentionKernel; }
    const char* GetName() const { return "SplitAttention"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
    size_t GetInputTypeCount() const { return 5; }
    ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
};
} // namespace pocket_accel
#endif
