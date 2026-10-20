// Standalone legacy English CPU benchmark; see V1_EXPERIMENTAL.md.
#include "../json.hpp"
#include "../ptts_engine.cpp"

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: benchmark MODELS VOICE int8|fp32 TEXT REPORT_JSON WARM_WAV\n";
        return 2;
    }
    try {
        pocket_tts::Config cfg;
        cfg.models_dir = argv[1];
        cfg.tokenizer_path = cfg.models_dir + "/tokenizer.model";
        cfg.precision = argv[3];
        cfg.temperature = 0;
        cfg.seed = 42;
        cfg.ar_threads = 1;
        cfg.dec_threads = 3;
        cfg.first_chunk_frames = 1;
        cfg.max_chunk_frames = 4;
        cfg.voice_cache = false;
        using Clock = std::chrono::steady_clock;
        auto now = Clock::now();
        auto ms = [](Clock::time_point start) {
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        };
        pocket_tts::PocketTTS engine(cfg);
        nlohmann::json report = {{"models", cfg.models_dir}, {"precision", cfg.precision},
            {"load_ms", ms(now)}, {"temperature", 0}, {"seed", 42},
            {"ar_threads", 1}, {"decoder_threads", 3}, {"max_chunk", 4},
            {"runs", nlohmann::json::array()}};
        for (int run = 0; run < 4; ++run) {
            size_t samples = 0, callbacks = 0;
            double ttfa = 0;
            std::vector<float> audio;
            now = Clock::now();
            engine.stream(argv[4], argv[2], [&](const float* pcm, size_t n) {
                if (callbacks++ == 0) ttfa = ms(now);
                samples += n;
                if (run == 1) audio.insert(audio.end(), pcm, pcm + n);
                return true;
            });
            const double elapsed = ms(now);
            if (samples == 0) throw std::runtime_error("No audio generated");
            report["runs"].push_back({{"run", run}, {"cold", run == 0},
                {"ttfa_ms", ttfa}, {"audio_seconds", double(samples)/24000},
                {"elapsed_ms", elapsed}, {"rtfx", double(samples)/24/elapsed}});
            if (run == 1) pocket_tts::PocketTTS::save_audio({std::move(audio), 24000}, argv[6]);
        }
        std::ofstream output(argv[5]);
        output << report.dump(2) << '\n';
        if (!output) throw std::runtime_error("Could not write report");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
