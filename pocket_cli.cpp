// PocketTTS CLI executable
// Refactored from the original single-file implementation.

#include "pocket_tts_engine.cpp"

#ifndef PTT_SHARED_LIB

int main(int argc, char* argv[]) {
    pocket_tts::Config cfg;
    bool stdout_output = false;
    std::string text, voice, output;
    int pos = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> char* {
            if (++i >= argc) { std::cerr << "Missing value for " << a << "\n"; std::exit(1); }
            return argv[i];
        };

        if (a == "-h" || a == "--help") {
            std::cerr
                << "Usage: " << argv[0] << " [OPTIONS] TEXT VOICE [OUTPUT]\n\n"
                << "Options:\n"
                << "  --precision <int8|fp32>  Model precision (default: int8)\n"
                << "  --temperature <float>    Sampling temperature (default: 0.7)\n"
                << "  --lsd-steps <int>        Flow matching steps (default: 1)\n"
                << "  --threads <int>          Total thread budget (default: 0 = half cores)\n"
                << "  --models-dir <path>      ONNX models directory (default: models)\n"
                << "  --voices-dir <path>      Voice samples directory (default: voices)\n"
                << "  --tokenizer <path>       Tokenizer path (default: models/tokenizer.model)\n"
                << "  --eos-threshold <float>  EOS detection threshold (default: -4.0)\n"
                << "  --noise-clamp <float>    Noise clamp value (default: 0, disabled)\n"
                << "  --eos-extra <int>        Extra frames after EOS (default: -1, auto)\n"
                << "  --first-chunk <int>      Frames in first decode chunk (default: 1)\n"
                << "  --max-chunk <int>        Max frames per decode chunk (default: 15)\n"
                << "  --no-cache               Disable all disk caching (.emb and .kv files)\n"
                << "  --stdout                 Output raw f32le PCM to stdout\n"
                << "  --verbose                Enable verbose output\n"
                << "  --profile                Show profiling report with first chunk latency\n";
            return 0;
        }
        else if (a == "--precision") cfg.precision = next();
        else if (a == "--temperature") cfg.temperature = std::stof(next());
        else if (a == "--lsd-steps") cfg.lsd_steps = std::stoi(next());
        else if (a == "--threads") cfg.num_threads = std::stoi(next());
        else if (a == "--models-dir") cfg.models_dir = next();
        else if (a == "--voices-dir") cfg.voices_dir = next();
        else if (a == "--tokenizer") cfg.tokenizer_path = next();
        else if (a == "--eos-threshold") cfg.eos_threshold = std::stof(next());
        else if (a == "--noise-clamp") cfg.noise_clamp = std::stof(next());
        else if (a == "--eos-extra") cfg.eos_extra_frames = std::stoi(next());
        else if (a == "--first-chunk") cfg.first_chunk_frames = std::stoi(next());
        else if (a == "--max-chunk") cfg.max_chunk_frames = std::stoi(next());
        else if (a == "--no-cache") cfg.voice_cache = false;
        else if (a == "--stdout") stdout_output = true;
        else if (a == "--verbose") cfg.verbose = true;
        else if (a == "--profile") pocket_tts::g_prof.enabled = true;
        else if (!a.empty() && a[0] == '-') { std::cerr << "Unknown: " << a << "\n"; return 1; }
        else {
            if (pos == 0) text = a;
            else if (pos == 1) voice = a;
            else if (pos == 2) output = a;
            ++pos;
        }
    }

    if (pos < 2) {
        std::cerr << "Need: TEXT VOICE [OUTPUT]\n";
        return 1;
    }
    if (pos < 3 && !stdout_output) {
        std::cerr << "Need OUTPUT file (or use --stdout)\n";
        return 1;
    }
    if (stdout_output) {
        cfg.verbose = false;
        pocket_tts::g_prof.enabled = false;
    }

    try {
        int threads = cfg.num_threads ? cfg.num_threads : std::max(2, int(std::thread::hardware_concurrency()) / 2);

        if (!stdout_output) {
            std::cerr << "Loading (precision=" << cfg.precision << ", threads=" << threads << ")...\n";
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        pocket_tts::PocketTTS tts(cfg);
        auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count(); };

        if (!stdout_output) {
            std::cerr << "  Loaded in " << std::fixed << std::setprecision(2) << elapsed() << "s\n";
            std::cerr << "Generating: \"" << text << "\" with " << voice << "\n";
        }
        t0 = std::chrono::high_resolution_clock::now();

        pocket_tts::AudioData audio;
        double first_chunk_latency = 0;

        if (stdout_output) {
#ifdef _WIN32
            _setmode(_fileno(stdout), _O_BINARY);
#endif
            size_t total_samples = 0;
            bool first = true;
            tts.stream(text, voice, [&](const float* s, size_t n) {
                if (first) {
                    first_chunk_latency = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                    first = false;
                }
                fwrite(s, sizeof(float), n, stdout);
                fflush(stdout);
                total_samples += n;
                return true;
            });
            audio.sample_rate = pocket_tts::PocketTTS::SR;
            audio.samples.resize(total_samples);
        } else {
            if (pocket_tts::g_prof.enabled) {
                std::vector<float> samples;
                bool first = true;
                tts.stream(text, voice, [&](const float* s, size_t n) {
                    if (first) {
                        first_chunk_latency = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t0).count();
                        first = false;
                    }
                    samples.insert(samples.end(), s, s + n);
                    return true;
                });
                audio = {std::move(samples), pocket_tts::PocketTTS::SR};
            } else {
                audio = tts.generate(text, voice);
            }
        }

        double gen_time = elapsed();
        double duration = audio.duration_sec();

        std::cerr << "  " << std::fixed << std::setprecision(2)
                  << duration << "s audio in " << gen_time << "s (RTFx: " << duration / gen_time << "x)\n";
        if (pocket_tts::g_prof.enabled) {
            std::cerr << "  First chunk latency: " << std::fixed << std::setprecision(0)
                      << first_chunk_latency << "ms\n";
        }

        if (!stdout_output) {
            pocket_tts::PocketTTS::save_audio(audio, output);
            std::cerr << "  Saved: " << output << "\n";
        }

        if (pocket_tts::g_prof.enabled) {
            tts.print_profiling_report();
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

#endif // PTT_SHARED_LIB
