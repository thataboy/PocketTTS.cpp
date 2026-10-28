// Python-side validation uses the hosting ONNX Runtime's API table.
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include "custom_ops.h"

extern "C" __attribute__((visibility("default")))
OrtStatus* RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* base) {
    const OrtApi* api = base->GetApi(ORT_API_VERSION);
    if (!api) return base->GetApi(1)->CreateStatus(ORT_FAIL, "Custom ops require a compatible ONNX Runtime version");
    Ort::InitApi(api);
    try {
        return api->AddCustomOpDomain(options, pocket_accel::domain());
    } catch (const std::exception& error) {
        return api->CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
    }
}
