// Warm, in-process benchmark; build with the optional Accelerate CMake target.
#include "../ptts_engine.cpp"
#ifdef __APPLE__
#include <mach/mach.h>
#endif

static int benchmark_main(const char* path) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "attention-benchmark");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    pocket_accel::register_ops(options);
    pocket_tts::OrtSession session(env, path, options);
    pocket_tts::StatefulRunner runner(session, false, true);
    nlohmann::json rows = nlohmann::json::array();
    for (int history : {128, 384, 768}) {
        std::vector<float> text(size_t(history)*1024);
        for (size_t i = 0; i < text.size(); ++i) text[i] = 0.1f*std::sin(float(i)*0.013f);
        const int64_t text_shape[] = {1, history, 1024}, empty_seq[] = {1, 0, 32};
        const int64_t empty_text[] = {1, 0, 1024}, sequence_shape[] = {1, 1, 32};
        for (int repeat = 0; repeat < 4; ++repeat) {
            runner.reset_state();
            std::vector<Ort::Value> prefill;
            prefill.push_back(Ort::Value::CreateTensor<float>(runner.mem(), nullptr, 0, empty_seq, 3));
            prefill.push_back(Ort::Value::CreateTensor<float>(runner.mem(), text.data(), text.size(), text_shape, 3));
            runner.run(prefill);
            std::vector<float> latent(32);
            auto start = std::chrono::steady_clock::now();
            for (int frame = 0; frame < 96; ++frame) {
                for (int d = 0; d < 32; ++d) latent[d] = 0.1f*std::sin(float(frame*32+d)*0.017f);
                std::vector<Ort::Value> inputs;
                inputs.push_back(Ort::Value::CreateTensor<float>(runner.mem(), latent.data(), 32, sequence_shape, 3));
                inputs.push_back(Ort::Value::CreateTensor<float>(runner.mem(), nullptr, 0, empty_text, 3));
                runner.run(inputs);
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            rows.push_back({{"history", history}, {"repeat", repeat}, {"seconds", seconds}, {"frames", 96}});
        }
    }
    std::cout << rows.dump(2) << '\n';
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--main") return benchmark_main(argv[2]);
    if (argc != 8) return 2; // bundle, voice, AR threads, decoder threads, chunk, text file, repeats
    pocket_tts::Config cfg;
    cfg.models_dir = argv[1];
    cfg.precision = "int8";
    cfg.ar_threads = std::stoi(argv[3]);
    cfg.dec_threads = std::stoi(argv[4]);
    cfg.max_chunk_frames = std::stoi(argv[5]);
    cfg.temperature = 0.3f;
    cfg.seed = 42;
    cfg.voice_cache = false;
    std::ifstream text_file(argv[6]);
    if (!text_file) return 2;
    std::string text((std::istreambuf_iterator<char>(text_file)), {});
    pocket_tts::PocketTTS engine(cfg);
    nlohmann::json rows = nlohmann::json::array();
    for (int run = 0; run < std::stoi(argv[7]); ++run) {
        pocket_tts::rng::seed(42);
        auto start = std::chrono::steady_clock::now();
        auto seconds = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(); };
        size_t count = 0;
        double first = -1;
        engine.stream(text, argv[2], [&](const float*, size_t n) {
            if (first < 0) first = seconds() * 1000;
            count += n;
            return true;
        });
        const double elapsed = seconds();
        nlohmann::json row = {{"run", run}, {"seconds", elapsed}, {"samples", count},
            {"audio_seconds", double(count)/24000}, {"rtfx", double(count)/24000/elapsed}, {"ttfa_ms", first}};
#ifdef __APPLE__
        mach_task_basic_info info;
        mach_msg_type_number_t size = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &size) == KERN_SUCCESS)
            row["rss_mib"] = double(info.resident_size)/(1024*1024);
#endif
        rows.push_back(row);
    }
    std::cout << rows.dump(2) << '\n';
}
