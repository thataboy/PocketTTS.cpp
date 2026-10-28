#pragma once
#include "decoder_conv.h"
#include "ar_attention.h"

#ifdef PTT_ACCELERATE
namespace pocket_accel {
inline Ort::CustomOpDomain& domain() {
    static ConvOp<false> conv;
    static ConvOp<true> conv_elu;
    static AttentionOp attention;
    static Ort::CustomOpDomain ops = [] {
        Ort::CustomOpDomain d("pockettts");
        d.Add(&conv);
        d.Add(&conv_elu);
        d.Add(&attention);
        return d;
    }();
    return ops;
}
inline void register_ops(Ort::SessionOptions& options) { options.Add(domain()); }
} // namespace pocket_accel
#endif
