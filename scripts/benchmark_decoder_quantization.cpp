// Exploratory warm-engine benchmark; production engine is compiled unchanged.
#include "../ptts_engine.cpp"

int main(int argc, char** argv) {
    if (argc != 3) return 2; // bundle, voice WAV
    pocket_tts::Config cfg;
    cfg.models_dir = argv[1];
    cfg.precision = "int8"; // AR/sampler INT8; decoder comes from the manifest.
    cfg.temperature = 0.3f;
    cfg.ar_threads = 1;
    cfg.dec_threads = 3;
    cfg.max_chunk_frames = 4;
    cfg.voice_cache = false;
    cfg.seed = 42;
    auto seconds = [](auto start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    };
    auto start = std::chrono::steady_clock::now();
    pocket_tts::PocketTTS engine(cfg);
    nlohmann::json report = {{"load_s", seconds(start)}, {"runs", nlohmann::json::array()}};
    const std::vector<std::string> texts = {
        "The morning sun lit the quiet street. A narrator paused before telling the next part of the story.",
        "Hello, world. This is a test of the new speech model.",
        "She was born in New York in 1943. Her father was a Hungarian Jewish immigrant."
    };
    for (int run = 0; run < 7; ++run) {
        // Identical AR noise/latents across decoder candidates, regardless of speed.
        pocket_tts::rng::seed(42+run);
        start = std::chrono::steady_clock::now();
        double first_ms = -1;
        size_t samples = 0;
        engine.stream(texts[run % texts.size()], argv[2], [&](const float*, size_t n) {
            if (first_ms < 0) first_ms = seconds(start)*1000;
            samples += n;
            return true;
        });
        double elapsed = seconds(start);
        report["runs"].push_back({{"stage", run == 0 ? "cold" : "warm"}, {"run",run},
            {"audio_s", double(samples)/24000}, {"generation_s",elapsed},
            {"rtfx",double(samples)/24000/elapsed}, {"ttfa_ms",first_ms}});
    }
    std::cout << report.dump(2) << '\n';
}
