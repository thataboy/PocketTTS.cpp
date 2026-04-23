// PocketTTS.cpp — Single-file C++ TTS runtime using ONNX Runtime
// https://github.com/VolgaGerm/PocketTTS.cpp
//
// Build with CMake:
//   cmake -B .build -DCMAKE_BUILD_TYPE=Release
//   cmake --build .build -j$(nproc)

#include <cctype>
#ifdef _WIN32
  #define ptt_mkdir(path) _mkdir(path)
#else
  #define ptt_mkdir(path) mkdir(path, 0755)
#endif

#include <sys/stat.h>
#include <csignal>

// ── External Libraries ──────────────────────────────────────────────────────

#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

// ── Standard Library ────────────────────────────────────────────────────────

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <dirent.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace pocket_tts {

// ════════════════════════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════════════════════════

static size_t calc_numel(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    size_t n = 1;
    for (auto d : shape) n *= (d > 0 ? d : 1);
    return n;
}

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;

    Tensor() = default;
    Tensor(std::vector<int64_t> s) : shape(std::move(s)), data(calc_numel(shape), 0.0f) {}
    Tensor(std::vector<float> d, std::vector<int64_t> s) : shape(std::move(s)), data(std::move(d)) {}

    size_t numel() const { return data.size(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }

    Tensor& reshape(std::vector<int64_t> ns) {
        int64_t neg = -1, known = 1;
        for (size_t i = 0; i < ns.size(); ++i) {
            if (ns[i] == -1) neg = i;
            else known *= ns[i];
        }
        if (neg >= 0) ns[neg] = numel() / known;
        shape = std::move(ns);
        return *this;
    }

    Tensor squeeze(int64_t dim = -1) const {
        std::vector<int64_t> ns;
        for (size_t i = 0; i < shape.size(); ++i)
            if (shape[i] != 1 || (dim >= 0 && (int64_t)i != dim)) ns.push_back(shape[i]);
        if (ns.empty()) ns.push_back(1);
        return Tensor(data, ns);
    }

    static Tensor concat(const std::vector<Tensor>& ts, int64_t dim) {
        if (ts.empty()) throw std::runtime_error("Cannot concat empty list");
        if (dim < 0) dim += ts[0].shape.size();

        std::vector<int64_t> os = ts[0].shape;
        int64_t total = 0;
        for (const auto& t : ts) total += t.shape[dim];
        os[dim] = total;

        Tensor r(os);
        int64_t outer = 1, inner = 1;
        for (int64_t i = 0; i < dim; ++i) outer *= os[i];
        for (size_t i = dim + 1; i < os.size(); ++i) inner *= os[i];

        int64_t off = 0;
        for (const auto& t : ts) {
            int64_t td = t.shape[dim], chunk = td * inner;
            for (int64_t o = 0; o < outer; ++o)
                std::memcpy(r.data.data() + o * total * inner + off * inner,
                           t.data.data() + o * chunk, chunk * sizeof(float));
            off += td;
        }
        return r;
    }

};

struct TensorI64 {
    std::vector<int64_t> shape;
    std::vector<int64_t> data;
    TensorI64() = default;
    TensorI64(std::vector<int64_t> s) : shape(std::move(s)), data(calc_numel(shape), 0) {}
    size_t numel() const { return data.size(); }
    int64_t* ptr() { return data.data(); }
    const int64_t* ptr() const { return data.data(); }
};

struct Config {
    std::string models_dir = "models", tokenizer_path = "models/tokenizer.model";
    std::string voices_dir = "voices";
    std::string precision = "int8";
    float temperature = 0.7f;
    float eos_threshold = -4.0f;
    float noise_clamp = 0.0f;
    int lsd_steps = 1, num_threads = 0, first_chunk_frames = 1, max_chunk_frames = 15;
    int eos_extra_frames = -1;  // -1 = auto-calculate from text length
    bool verbose = false;
    bool voice_cache = true;
    bool refresh_cache = false;
};

struct AudioData {
    std::vector<float> samples;
    int sample_rate = 24000;
    float duration_sec() const { return float(samples.size()) / sample_rate; }
};

using StreamCallback = std::function<bool(const float*, size_t)>;

// ════════════════════════════════════════════════════════════════════════════
// Utilities
// ════════════════════════════════════════════════════════════════════════════

// ── RNG (xoshiro256**) ──────────────────────────────────────────────────────

namespace rng {
static uint64_t s[4] = {0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL, 0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t next() {
    uint64_t result = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
}

void seed(uint64_t v) {
    for (int i = 0; i < 4; ++i) {
        v += 0x9E3779B97F4A7C15ULL;
        uint64_t z = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        s[i] = z ^ (z >> 31);
    }
}

static float uniform() { return (next() >> 11) * (1.0f / 9007199254740992.0f); }

float normal(float mean = 0, float stddev = 1) {
    float u1 = uniform(), u2 = uniform();
    while (u1 <= 1e-10f) u1 = uniform();
    return mean + stddev * std::sqrt(-2.0f * std::log(u1)) * std::cos(6.283185307179586f * u2);
}

void fill_normal(float* data, size_t n, float mean = 0, float stddev = 1) {
    for (size_t i = 0; i < n; ++i) data[i] = normal(mean, stddev);
}
} // namespace rng

// ── Audio Resampling (Lanczos) ──────────────────────────────────────────────

static std::vector<float> resample(const std::vector<float>& in, int src, int dst) {
    if (src == dst) return in;
    constexpr int K = 16;
    constexpr float PI = 3.14159265358979323846f;
    double ratio = double(dst) / src;
    std::vector<float> out(size_t(in.size() * ratio));

    auto sinc = [&](float x) { return std::abs(x) < 1e-6f ? 1.0f : std::sin(PI * x) / (PI * x); };
    auto lanczos = [&](float x) { return std::abs(x) >= K ? 0.0f : sinc(x) * sinc(x / K); };

    for (size_t i = 0; i < out.size(); ++i) {
        double sp = i / ratio;
        int64_t c = int64_t(sp);
        float f = float(sp - c), sample = 0, wsum = 0;
        for (int k = -K + 1; k <= K; ++k) {
            int64_t idx = c + k;
            if (idx >= 0 && idx < int64_t(in.size())) {
                float w = lanczos(k - f);
                sample += in[idx] * w;
                wsum += w;
            }
        }
        out[i] = wsum > 0 ? sample / wsum : 0;
    }
    return out;
}

// ── Sentence Splitting ──────────────────────────────────────────────────────
// Splits on sentence-ending punctuation .? followed by whitespace or EOF.
// Preserves punctuation with the sentence. Handles common abbreviations.

// Splits on fallback punctuation ,;:)] and --
//  if sentence length passes soft limit and .? has not been found yet.
// Splits on whitespace if length passes hard limit
// Loosely enforces minimum sentence length for better prosody.
// Does not split at all if sentence is under soft limit.

// normalize whitespace; remove troublesome chars
static std::string clean_text(const std::string& raw, bool keep_newl) {
    std::string text;
    text.reserve(raw.size());

    bool prev_space = true; // skip leading ' ' \t'
    bool prev_newl = true;  // skip leading \n
    bool has_alnum = false;

    for (size_t i = 0; i < raw.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(raw[i]);

        // Whitespace collapsing
        if (c == ' ' || c == '\t') {
            if (!prev_space) {
                text += ' ';
                prev_space = true;
            }
            continue;
        }
        if (c == '\r') continue;
        if (c == '\n') {
            if (!prev_newl) {
                text += keep_newl ? "\n" : ", ";
                prev_newl = prev_space = true;
            }
            continue;
        }

        // Handle special multi-byte UTF-8 sequences
        if (c == 0xE2 && i + 2 < raw.size()) {
            unsigned char c2 = static_cast<unsigned char>(raw[i + 1]);

            if (c2 == 0x80) {
                unsigned char c3 = static_cast<unsigned char>(raw[i + 2]);

                if (c3 == 0x9C || c3 == 0x9D || c3 == 0x98) { // strip “ ” ‘
                    i += 2; continue;
                }
                if (c3 == 0x99) { // right curly ’
                    // change to ' if surrounded by alphas, otherwise strip
                    if (i > 0
                        && std::isalpha(static_cast<unsigned char>(raw[i-1]))
                        && i+3 < raw.size()
                        && std::isalpha(static_cast<unsigned char>(raw[i+3]))) {
                        text += '\'';
                    }
                    i += 2; continue;
                }
                if (c3 == 0x94) { // emdash — becomes --
                    text += "--";
                    prev_space = prev_newl = false;
                    i += 2; continue;
                }
                if (c3 == 0xA6) { // ellipsis … becomes ...
                    text += "...";
                    prev_space = prev_newl = false;
                    i += 2; continue;
                }
            }
        }

        if (c == '!') c = '.'; // keep Pocket from shouting
        else if (c == '\'') {
            // only keep ' if between two alphas
            if (i == 0 || i + 1 >= raw.size()
                || !std::isalpha(static_cast<unsigned char>(raw[i-1]))
                || !std::isalpha(static_cast<unsigned char>(raw[i+1])))
            continue;
        }
        else if (c == '"' || c == '~' || c == '`') {  // Skip these entirely
            continue;
        }
        text += c;
        has_alnum |= std::isalnum(c);
        prev_newl = prev_space = false;
    }
    return has_alnum ? text : "";
}

static std::vector<std::string> split_sentences(const std::string& raw) {
    constexpr size_t kHardLimit = 350;
    constexpr size_t kSoftLimit = 300;
    constexpr size_t kMinLen = 75;

    // for long text, clean_text should keep \n to use as hard break points
    std::string text = clean_text(raw, raw.size() > kSoftLimit);
    std::vector<std::string> sentences;

    auto push_chunk = [&](const std::string& chunk) {
        // push only if non empty
        size_t end = chunk.find_last_not_of(" ");
        if (end != std::string::npos) {
            // ending sentence with punctuation helps prevent weird noises
            if (std::ispunct(static_cast<unsigned char>(chunk[end])))
                sentences.push_back(chunk);
            else
                sentences.push_back(chunk + '.');
        }
    };

    auto push_final_chunk = [&](const std::string& chunk) {
        push_chunk(chunk);
        // add pseudo silence if sentences is empty
        if (sentences.empty()) sentences.push_back(";");
    };

    if (text.size() <= kSoftLimit) {
        push_final_chunk(text);
        return sentences;
    }

    std::string current;
    current.reserve(kHardLimit);

    // Using position + 1 for trackers to make code simpler; 0 == not found
    // position + 1 of last seen primary break character .?
    //   that we passed up because current is too short
    size_t last_primary_split = 0;
    // position + 1 of last seen fallback break character ,;:)]--
    //   used when soft limit is exceeded
    size_t last_fallback_split = 0;
    // position + 1 of last seen space
    //   used as last resort if no primary or fallback break found
    size_t last_space_split = 0;

    auto is_abbreviation = [&](const std::string& s, size_t dot_pos) -> bool {
        if (dot_pos < 2) return false;
        size_t start = dot_pos;
        while (start > 0 && std::isalpha((unsigned char)s[start - 1])) start--;
        std::string word = s.substr(start, dot_pos - start);
        for (auto& c : word) c = std::tolower((unsigned char)c);
        return word == "mr" || word == "mrs" || word == "ms" || word == "dr" ||
               word == "st" || word == "jr" || word == "sr" || word == "vs" ||
               word == "etc" || word == "inc" || word == "ltd" || word == "prof" ||
               word == "gen" || word == "gov" || word == "sgt" || word == "cpl" ||
               word == "pvt" || word == "capt" || word == "lt" || word == "col";
    };

    auto is_at_boundary = [&](const std::string& s, size_t i) -> bool {
        if (++i >= s.size()) return true;
        return !std::isalnum(static_cast<unsigned char>(s[i]));
    };

    auto is_primary_break = [&](const std::string& s, size_t i) -> bool {
        char ch = s[i];
        if (ch != '.' && ch != '?') return false;
        if (ch == '.') {
            if (i + 1 < s.size() && s[i + 1] == '.') return false;
            if (i > 0 && s[i - 1] == '.') return false;
            if (is_abbreviation(s, i)) return false;
        }
        return is_at_boundary(s, i);
    };

    auto is_fallback_break = [&](const std::string& s, size_t i) -> bool {
        if (std::string(",;:)]").find(s[i]) != std::string::npos) {
            return is_at_boundary(s, i);
        }
        // break on --
        if (s[i] == '-' && i > 0 && s[i-1] == '-' && (i+1 >= s.size() || s[i+1] != '-')) {
            return is_at_boundary(s, i);
        }
        return false;
    };

    // Consume trailing )]} that should stay with punctuation
    auto consume_closers = [&](const std::string& s, size_t& i, std::string& current) {
        while (i + 1 < s.size()) {
            unsigned char next = s[i + 1];
            if (next == ')' || next == ']' || next == '}') {
                ++i;
                current += s[i];
            } else {
                break;
            }
        }
    };

    auto push_chunk_and_reset = [&](const std::string& chunk) {
        push_chunk(chunk);
        current.clear();
        last_fallback_split = last_primary_split = last_space_split = 0;
    };

    // main loop
    for (size_t i = 0; i < text.size(); ++i) {

        // hard split on \n  do it here so \n is not included
        if (text[i] == '\n') {
            push_chunk_and_reset(current);
            continue;
        }

        current += text[i];

        if (text[i] == ' ') {
            last_space_split = current.size();

        } else if (is_primary_break(text, i)) {
            consume_closers(text, i, current);
            if (current.size() >= kMinLen) {
                push_chunk_and_reset(current);
                continue;
            }
            last_primary_split = current.size();

        } else if (is_fallback_break(text, i)) {
            consume_closers(text, i, current);
            last_fallback_split = current.size();
        }

        if (current.size() >= kSoftLimit) {
            size_t split_size = std::max(last_primary_split, last_fallback_split);

            if (!split_size && current.size() >= kHardLimit) {
                // last resort: split on last space or arbitrarily at current position
                split_size = last_space_split ? last_space_split : current.size();
            }

            if (split_size) {
                // backtrack i to scan rest of current
                i -= current.size() - split_size;
                push_chunk_and_reset(current.substr(0, split_size));
            }
        }
    }

    push_final_chunk(current);
    return sentences;
}

// compute frames_after_eos.
static int calc_eos_extra(const std::string& text, int eos_extra) {

    auto count_words = [&](const std::string& text, int max=INT_MAX) {
        int count = 0;
        bool in_word = false;
        for (char c : text) {
            if (c == ' ') { in_word = false; }
            else if (!in_word) {
                in_word = true;
                if (++count >= max) return count;
            }
        }
        return count;
    };
    if (eos_extra < 0) {
        int nwords = count_words(text, 5);
        uint64_t val = rng::next();
        if (nwords < 5) {
            eos_extra = 3;
        } else {
            char c = text.back();
            eos_extra =
                c == '.' || c == '?' || c == ';'
                ? 3 + (val % 3)
                : 2 + (val % 2);
        }
    }

    std::cerr << "〔" << text << "〕\n";
    return eos_extra;
}

// ════════════════════════════════════════════════════════════════════════════
// Profiler
// ════════════════════════════════════════════════════════════════════════════

struct Profiler {
    struct Timer {
        std::string name;
        double total_ms = 0;
        int count = 0;
        double min_ms = 1e9, max_ms = 0;

        void add(double ms) {
            total_ms += ms;
            count++;
            min_ms = std::min(min_ms, ms);
            max_ms = std::max(max_ms, ms);
        }
        double avg_ms() const { return count > 0 ? total_ms / count : 0; }
    };

    std::unordered_map<std::string, Timer> timers;
    bool enabled = false;

    class ScopedTimer {
        Profiler& prof;
        std::string name;
        std::chrono::high_resolution_clock::time_point start;
    public:
        ScopedTimer(Profiler& p, const std::string& n) : prof(p), name(n), start(std::chrono::high_resolution_clock::now()) {}
        ~ScopedTimer() {
            if (prof.enabled) {
                auto end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(end - start).count();
                prof.timers[name].name = name;
                prof.timers[name].add(ms);
            }
        }
    };

    ScopedTimer time(const std::string& name) { return ScopedTimer(*this, name); }

    void report() const {
        std::cout << "\n========== PROFILING REPORT ==========\n";
        std::vector<std::pair<std::string, Timer>> sorted;
        for (const auto& [k, v] : timers) sorted.emplace_back(k, v);
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second.total_ms > b.second.total_ms; });

        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::left << std::setw(35) << "Operation"
                  << std::right << std::setw(10) << "Total(ms)"
                  << std::setw(10) << "Count"
                  << std::setw(10) << "Avg(ms)"
                  << std::setw(10) << "Min(ms)"
                  << std::setw(10) << "Max(ms)" << "\n";
        std::cout << std::string(85, '-') << "\n";

        for (const auto& [name, t] : sorted) {
            std::cout << std::left << std::setw(35) << name
                      << std::right << std::setw(10) << t.total_ms
                      << std::setw(10) << t.count
                      << std::setw(10) << t.avg_ms()
                      << std::setw(10) << t.min_ms
                      << std::setw(10) << t.max_ms << "\n";
        }
        std::cout << "=======================================\n";
    }

    void reset() { timers.clear(); }
};

static Profiler g_prof;

// ════════════════════════════════════════════════════════════════════════════
// Disk Cache
//
// Two layers of on-disk caching, both stored under voices/.cache/:
//
//   .emb files — Mimi encoder output (voice embedding).
//                Avoids re-encoding the same WAV file on every run.
//
//   .kv files  — Transformer KV state after voice conditioning.
//                Avoids re-running the expensive voice conditioning pass.
//                On cache hit, restoring a KV snapshot takes ~4ms vs
//                hundreds of ms for a full conditioning pass.
// ════════════════════════════════════════════════════════════════════════════

namespace cache {

static time_t get_mtime(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return 0;
    return st.st_mtime;
}

static bool mkdir_p(const std::string& path) {
    size_t pos = 0;
    while (pos < path.size()) {
        size_t slash = path.find('/', pos + 1);
        size_t bslash = path.find('\\', pos + 1);
        pos = std::min(slash, bslash);
        if (pos == std::string::npos) break;
        std::string sub = path.substr(0, pos);
        if (!sub.empty()) ptt_mkdir(sub.c_str());
    }
    return ptt_mkdir(path.c_str()) == 0 || errno == EEXIST;
}

// Derive cache file path: voices/.cache/{stem}.{ext}
// ext = "emb" for voice embeddings, "kv" for KV state snapshots
static std::string get_cache_path(const std::string& voices_dir, const std::string& voice_path, const char* ext = "emb") {
    std::string filename = voice_path;
    size_t slash = voice_path.find_last_of("/\\");
    if (slash != std::string::npos) filename = voice_path.substr(slash + 1);
    size_t dot = filename.rfind('.');
    if (dot != std::string::npos) filename = filename.substr(0, dot);
    return voices_dir + "/.cache/" + filename + "." + ext;
}

static bool is_cache_valid(const std::string& voice_path, const std::string& cache_path) {
    time_t voice_mtime = get_mtime(voice_path);
    time_t cache_mtime = get_mtime(cache_path);
    return cache_mtime > 0 && cache_mtime >= voice_mtime;
}

// ── Voice Embedding (.emb) Format ───────────────────────────────────────────
// [4B magic "EMB1"] [4B ndims] [ndims*8B shape] [numel*4B float data]

static constexpr uint32_t EMB_MAGIC = 0x31424D45; // "EMB1" little-endian

static bool save_embedding(const std::string& path, const std::vector<int64_t>& shape, const std::vector<float>& data) {
    size_t slash = path.find_last_of('/');
    if (slash != std::string::npos) {
        mkdir_p(path.substr(0, slash));
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    uint32_t magic = EMB_MAGIC;
    int32_t ndims = static_cast<int32_t>(shape.size());

    f.write(reinterpret_cast<const char*>(&magic), 4);
    f.write(reinterpret_cast<const char*>(&ndims), 4);
    f.write(reinterpret_cast<const char*>(shape.data()), ndims * sizeof(int64_t));
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));

    return f.good();
}

static bool load_embedding(const std::string& path, std::vector<int64_t>& shape, std::vector<float>& data) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    uint32_t magic;
    int32_t ndims;

    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != EMB_MAGIC) return false;

    f.read(reinterpret_cast<char*>(&ndims), 4);
    if (ndims <= 0 || ndims > 10) return false;

    shape.resize(ndims);
    f.read(reinterpret_cast<char*>(shape.data()), ndims * sizeof(int64_t));

    size_t numel = 1;
    for (int32_t i = 0; i < ndims; ++i) {
        if (shape[i] <= 0) return false;
        numel *= shape[i];
    }

    data.resize(numel);
    f.read(reinterpret_cast<char*>(data.data()), numel * sizeof(float));

    return f.good();
}

} // namespace cache

// ════════════════════════════════════════════════════════════════════════════
// ONNX Runtime Wrappers
// ════════════════════════════════════════════════════════════════════════════

static Ort::Env& get_ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pocket_tts");
    return env;
}

// Convert std::string path to ORTCHAR_T string (wchar_t on Windows, char elsewhere)
static std::basic_string<ORTCHAR_T> to_ort_path(const std::string& s) {
#ifdef _WIN32
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (n <= 0) throw std::runtime_error("Failed to widen path: " + s);
    std::wstring w(n - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    return w;
#else
    return s;
#endif
}

// ── OrtSession ──────────────────────────────────────────────────────────────
// Thin wrapper around Ort::Session that caches input/output names and shapes.

class OrtSession {
    Ort::Session sess_;
    std::vector<std::string> in_names_, out_names_;
    std::vector<const char*> in_ptrs_, out_ptrs_;
    std::vector<std::vector<int64_t>> in_shapes_;
    std::vector<ONNXTensorElementDataType> in_types_;
    std::string name_;

public:
    OrtSession(Ort::Env& env, const std::string& path, const Ort::SessionOptions& opts, const std::string& name = "")
        : sess_(env, to_ort_path(path).c_str(), opts), name_(name.empty() ? path : name) {
        Ort::AllocatorWithDefaultOptions alloc;

        size_t num_in = sess_.GetInputCount();
        for (size_t i = 0; i < num_in; ++i) {
            auto n = sess_.GetInputNameAllocated(i, alloc);
            in_names_.push_back(n.get());
            auto ti = sess_.GetInputTypeInfo(i);
            auto tsi = ti.GetTensorTypeAndShapeInfo();
            in_shapes_.push_back(tsi.GetShape());
            in_types_.push_back(tsi.GetElementType());
        }

        size_t num_out = sess_.GetOutputCount();
        for (size_t i = 0; i < num_out; ++i) {
            auto n = sess_.GetOutputNameAllocated(i, alloc);
            out_names_.push_back(n.get());
        }

        for (const auto& n : in_names_) in_ptrs_.push_back(n.c_str());
        for (const auto& n : out_names_) out_ptrs_.push_back(n.c_str());
    }

    Ort::Session& session() { return sess_; }

    std::vector<Ort::Value> run(const std::vector<Ort::Value>& in) {
        auto _ = g_prof.time("run:" + name_);
        return sess_.Run(Ort::RunOptions{nullptr}, in_ptrs_.data(), in.data(), in.size(), out_ptrs_.data(), out_ptrs_.size());
    }

    void run_with_binding(Ort::IoBinding& binding) {
        auto _ = g_prof.time("run:" + name_);
        sess_.Run(Ort::RunOptions{nullptr}, binding);
    }

    void print_info() const {
        std::cout << "\n  Model: " << name_ << "\n";
        std::cout << "    Inputs (" << in_names_.size() << "):\n";
        for (size_t i = 0; i < in_names_.size(); ++i) {
            std::cout << "      [" << i << "] " << in_names_[i] << " : ";
            std::cout << type_str(in_types_[i]) << " ";
            print_shape(in_shapes_[i]);
            std::cout << "\n";
        }
        std::cout << "    Outputs (" << out_names_.size() << "):\n";
        for (size_t i = 0; i < out_names_.size(); ++i) {
            std::cout << "      [" << i << "] " << out_names_[i] << "\n";
        }
    }

    static std::string type_str(ONNXTensorElementDataType t) {
        switch (t) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
            default: return "type(" + std::to_string(t) + ")";
        }
    }

    static void print_shape(const std::vector<int64_t>& shape) {
        std::cout << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            if (shape[i] < 0) std::cout << "?";
            else std::cout << shape[i];
        }
        std::cout << "]";
    }

    const std::string& name() const { return name_; }
    const std::vector<std::string>& input_names() const { return in_names_; }
    const std::vector<std::string>& output_names() const { return out_names_; }
    const std::vector<std::vector<int64_t>>& input_shapes() const { return in_shapes_; }
    const std::vector<ONNXTensorElementDataType>& input_types() const { return in_types_; }
};

// ── StateBufferIO ───────────────────────────────────────────────────────────
// Manages the stateful inputs/outputs of the autoregressive transformer.
//
// The flow_lm_main model has ~60 state tensors (KV cache layers) that must
// be fed back as inputs on each step. This struct:
//
//   1. Double-buffers all state tensors so the output of step N becomes the
//      input of step N+1 without copying (just swap the buffer index).
//   2. Handles mixed types (float32, int64, bool) across state tensors.
//   3. Supports both fixed-size and dynamic-size states.
//   4. Provides Snapshot (fast in-memory) and DiskSnapshot (serialized blob)
//      for caching voice-conditioned KV state across runs.

struct StateBufferIO {
    std::vector<std::vector<float>> f32[2];
    std::vector<std::vector<int64_t>> i64[2];
    std::vector<std::vector<uint8_t>> b8[2];
    std::vector<std::vector<uint16_t>> f16[2];  // fp16 KV caches
    std::vector<std::vector<int64_t>> shapes;
    std::vector<std::vector<int64_t>> init_shapes;
    std::vector<ONNXTensorElementDataType> types;
    std::vector<std::string> names;
    std::vector<bool> is_dynamic;
    int current_buf = 0;

    void init(OrtSession& s) {
        const auto& in_names = s.input_names();
        const auto& in_shapes = s.input_shapes();
        const auto& in_types = s.input_types();

        for (size_t i = 0; i < in_names.size(); ++i) {
            if (in_names[i].find("state_") != 0) continue;
            names.push_back(in_names[i]);

            std::vector<int64_t> sh;
            bool dynamic = false;
            for (auto d : in_shapes[i]) {
                if (d <= 0) dynamic = true;
                sh.push_back(d > 0 ? d : 0);
            }
            shapes.push_back(sh);
            types.push_back(in_types[i]);
            is_dynamic.push_back(dynamic);

            size_t sz = 1;
            for (auto d : sh) sz *= (d > 0 ? d : 1);
            size_t alloc = dynamic ? 0 : sz;

            for (int b = 0; b < 2; ++b) {
                if (in_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                    i64[b].push_back(std::vector<int64_t>(alloc, 0));
                    f32[b].push_back({}); f16[b].push_back({}); b8[b].push_back({});
                } else if (in_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
                    b8[b].push_back(std::vector<uint8_t>(alloc, 0));
                    f32[b].push_back({}); f16[b].push_back({}); i64[b].push_back({});
                } else if (in_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                    // Single-buffered: only buffer 0 is allocated.
                    // Both input and output bind to buffer 0, enabling in-place scatter.
                    f16[b].push_back(b == 0 ? std::vector<uint16_t>(alloc, 0) : std::vector<uint16_t>());
                    f32[b].push_back({}); i64[b].push_back({}); b8[b].push_back({});
                } else {
                    f32[b].push_back(std::vector<float>(alloc, 0.0f));
                    f16[b].push_back({}); i64[b].push_back({}); b8[b].push_back({});
                }
            }
        }

        init_shapes = shapes;
    }

    int in_buf() const { return current_buf; }
    int out_buf() const { return 1 - current_buf; }
    void swap() { current_buf = 1 - current_buf; }

    // Reset all state buffers to zero without freeing/reallocating.
    // Ideal for fixed-size state models (e.g. Mimi decoder) where the
    // buffer sizes never change between runs.
    void reset() {
        current_buf = 0;
        size_t n = names.size();
        for (size_t i = 0; i < n; ++i) {
            for (int b = 0; b < 2; ++b) {
                if (is_dynamic[i]) {
                    f32[b][i].clear(); f16[0][i].clear();
                    i64[b][i].clear(); b8[b][i].clear();
                } else {
                    std::fill(f32[b][i].begin(), f32[b][i].end(), 0.0f);
                    std::fill(f16[0][i].begin(), f16[0][i].end(), uint16_t(0));
                    std::fill(i64[b][i].begin(), i64[b][i].end(), int64_t(0));
                    std::fill(b8[b][i].begin(), b8[b][i].end(), uint8_t(0));
                }
            }
        }
    }

    Ort::Value create_input_value(size_t state_idx, Ort::MemoryInfo& mem) {
        auto t = types[state_idx];
        // FP16 KV caches use single-buffered mode (always buffer 0) to enable
        // in-place scatter — ORT skips the bulk copy when src == dst.
        int b = (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ? 0 : in_buf();
        auto& sh = shapes[state_idx];
        if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            return Ort::Value::CreateTensor<int64_t>(mem, i64[b][state_idx].data(), i64[b][state_idx].size(),
                                                      sh.data(), sh.size());
        } else if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
            return Ort::Value::CreateTensor<bool>(mem, reinterpret_cast<bool*>(b8[b][state_idx].data()),
                                                   b8[b][state_idx].size(), sh.data(), sh.size());
        } else if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            return Ort::Value::CreateTensor<Ort::Float16_t>(mem, reinterpret_cast<Ort::Float16_t*>(f16[0][state_idx].data()),
                                                             f16[0][state_idx].size(), sh.data(), sh.size());
        } else {
            return Ort::Value::CreateTensor<float>(mem, f32[b][state_idx].data(), f32[b][state_idx].size(),
                                                    sh.data(), sh.size());
        }
    }

    Ort::Value create_output_value(size_t state_idx, Ort::MemoryInfo& mem) {
        auto t = types[state_idx];
        int b = (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ? 0 : out_buf();
        auto& sh = shapes[state_idx];
        if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            return Ort::Value::CreateTensor<int64_t>(mem, i64[b][state_idx].data(), i64[b][state_idx].size(),
                                                      sh.data(), sh.size());
        } else if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
            return Ort::Value::CreateTensor<bool>(mem, reinterpret_cast<bool*>(b8[b][state_idx].data()),
                                                   b8[b][state_idx].size(), sh.data(), sh.size());
        } else if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            return Ort::Value::CreateTensor<Ort::Float16_t>(mem, reinterpret_cast<Ort::Float16_t*>(f16[0][state_idx].data()),
                                                             f16[0][state_idx].size(), sh.data(), sh.size());
        } else {
            return Ort::Value::CreateTensor<float>(mem, f32[b][state_idx].data(), f32[b][state_idx].size(),
                                                    sh.data(), sh.size());
        }
    }

    void copy_from_output(size_t state_idx, Ort::Value& val) {
        auto t = types[state_idx];
        int b = (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ? 0 : out_buf();
        auto info = val.GetTensorTypeAndShapeInfo();
        shapes[state_idx] = info.GetShape();
        size_t out_size = info.GetElementCount();

        if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            auto* src = val.GetTensorData<int64_t>();
            i64[b][state_idx].assign(src, src + out_size);
        } else if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
            auto* src = reinterpret_cast<const uint8_t*>(val.GetTensorData<bool>());
            b8[b][state_idx].assign(src, src + out_size);
        } else if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            auto* src = reinterpret_cast<const uint16_t*>(val.GetTensorData<Ort::Float16_t>());
            f16[0][state_idx].assign(src, src + out_size);
        } else {
            auto* src = val.GetTensorData<float>();
            f32[b][state_idx].assign(src, src + out_size);
        }
    }

    // ── DiskSnapshot ────────────────────────────────────────────────────────
    // Serialized blob for persisting KV state to disk (.kv files).
    // Format: [4B current_buf] [4B num_states] then per-state:
    //         [4B ndims] [ndims*8B shape] [4B type] [8B data_bytes] [data]

    struct DiskSnapshot {
        std::vector<uint8_t> blob;
        static constexpr uint32_t MAGIC = 0x3143564B;  // "KVC1" little-endian

        bool save_to_disk(const std::string& path) const {
            size_t slash = path.find_last_of('/');
            if (slash != std::string::npos) cache::mkdir_p(path.substr(0, slash));
            std::ofstream f(path, std::ios::binary);
            if (!f) return false;
            uint32_t magic = MAGIC;
            uint64_t sz = blob.size();
            f.write(reinterpret_cast<const char*>(&magic), 4);
            f.write(reinterpret_cast<const char*>(&sz), 8);
            f.write(reinterpret_cast<const char*>(blob.data()), blob.size());
            return f.good();
        }

        bool load_from_disk(const std::string& path) {
            std::ifstream f(path, std::ios::binary);
            if (!f) return false;
            uint32_t magic;
            uint64_t sz;
            f.read(reinterpret_cast<char*>(&magic), 4);
            if (magic != MAGIC) return false;
            f.read(reinterpret_cast<char*>(&sz), 8);
            if (sz == 0 || sz > 200 * 1024 * 1024) return false;
            blob.resize(sz);
            f.read(reinterpret_cast<char*>(blob.data()), sz);
            return f.good();
        }
    };

    // ── Snapshot ─────────────────────────────────────────────────────────────
    // Fast in-memory snapshot: all state data packed into contiguous buffers.
    // Restoring is a bulk memcpy into pre-sized buffers (~1ms for 60 states).

    struct Snapshot {
        std::vector<float> f32_data;
        std::vector<int64_t> i64_data;
        std::vector<uint8_t> b8_data;
        std::vector<uint16_t> f16_data;
        std::vector<size_t> f32_offsets;
        std::vector<size_t> i64_offsets;
        std::vector<size_t> b8_offsets;
        std::vector<size_t> f16_offsets;
        std::vector<std::vector<int64_t>> shapes;
        int current_buf;
    };

    Snapshot take_snapshot() const {
        Snapshot snap;
        int b = in_buf();
        size_t n = names.size();
        snap.shapes.resize(n);
        snap.current_buf = current_buf;

        // Detect sliceable KV cache states: large float32 or float16 buffers
        // paired with an int64 position counter at i+1 or i+2.
        struct SliceInfo { int seq_dim; int64_t used; };
        std::vector<SliceInfo> slices(n, {-1, -1});

        for (size_t i = 0; i < n; ++i) {
            bool is_f32 = types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && f32[b][i].size() >= 10000;
            bool is_f16 = types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 && f16[0][i].size() >= 10000;
            if (!is_f32 && !is_f16) continue;

            int seq_dim = -1;
            for (size_t d = 0; d < shapes[i].size(); ++d) {
                if (shapes[i][d] == 1000) { seq_dim = (int)d; break; }
            }
            if (seq_dim < 0) continue;

            for (size_t j = 1; j <= 2 && i + j < n; ++j) {
                if (types[i + j] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
                    i64[b][i + j].size() == 1 && i64[b][i + j][0] > 0 && i64[b][i + j][0] <= 1000) {
                    slices[i] = {seq_dim, i64[b][i + j][0]};
                    break;
                }
            }
        }

        // Compute total sizes with slicing
        size_t total_f32 = 0, total_i64 = 0, total_b8 = 0, total_f16 = 0;
        for (size_t i = 0; i < n; ++i) {
            if (slices[i].seq_dim >= 0) {
                auto sh = shapes[i];
                sh[slices[i].seq_dim] = slices[i].used;
                snap.shapes[i] = sh;
                size_t numel = 1;
                for (auto d : sh) numel *= (d > 0 ? d : 1);
                if (types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) total_f16 += numel;
                else total_f32 += numel;
            } else {
                snap.shapes[i] = shapes[i];
                total_f32 += f32[b][i].size();
                total_f16 += f16[0][i].size();
            }
            total_i64 += i64[b][i].size();
            total_b8 += b8[b][i].size();
        }

        snap.f32_data.resize(total_f32);
        snap.i64_data.resize(total_i64);
        snap.b8_data.resize(total_b8);
        snap.f16_data.resize(total_f16);
        snap.f32_offsets.resize(n + 1);
        snap.i64_offsets.resize(n + 1);
        snap.b8_offsets.resize(n + 1);
        snap.f16_offsets.resize(n + 1);

        size_t fo = 0, io = 0, bo = 0, ho = 0;
        for (size_t i = 0; i < n; ++i) {
            snap.f32_offsets[i] = fo;
            snap.i64_offsets[i] = io;
            snap.b8_offsets[i] = bo;
            snap.f16_offsets[i] = ho;

            if (slices[i].seq_dim >= 0) {
                int sd = slices[i].seq_dim;
                int64_t N = slices[i].used;
                int64_t outer = 1;
                for (int d = 0; d < sd; ++d) outer *= shapes[i][d];
                int64_t inner = 1;
                for (size_t d = sd + 1; d < shapes[i].size(); ++d) inner *= shapes[i][d];
                int64_t old_stride = shapes[i][sd] * inner;
                int64_t new_stride = N * inner;

                if (types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                    const uint16_t* src = f16[0][i].data();
                    uint16_t* dst = snap.f16_data.data() + ho;
                    for (int64_t o = 0; o < outer; ++o)
                        memcpy(dst + o * new_stride, src + o * old_stride, new_stride * sizeof(uint16_t));
                    ho += outer * new_stride;
                } else {
                    const float* src = f32[b][i].data();
                    float* dst = snap.f32_data.data() + fo;
                    for (int64_t o = 0; o < outer; ++o)
                        memcpy(dst + o * new_stride, src + o * old_stride, new_stride * sizeof(float));
                    fo += outer * new_stride;
                }
            } else {
                if (!f32[b][i].empty()) { memcpy(snap.f32_data.data() + fo, f32[b][i].data(), f32[b][i].size() * sizeof(float)); fo += f32[b][i].size(); }
                if (!f16[0][i].empty()) { memcpy(snap.f16_data.data() + ho, f16[0][i].data(), f16[0][i].size() * sizeof(uint16_t)); ho += f16[0][i].size(); }
            }

            if (!i64[b][i].empty()) { memcpy(snap.i64_data.data() + io, i64[b][i].data(), i64[b][i].size() * sizeof(int64_t)); io += i64[b][i].size(); }
            if (!b8[b][i].empty()) { memcpy(snap.b8_data.data() + bo, b8[b][i].data(), b8[b][i].size()); bo += b8[b][i].size(); }
        }
        snap.f32_offsets[n] = fo;
        snap.i64_offsets[n] = io;
        snap.b8_offsets[n] = bo;
        snap.f16_offsets[n] = ho;

        return snap;
    }

    void restore_snapshot(const Snapshot& snap) {
        current_buf = snap.current_buf;
        int b = in_buf();
        size_t n = names.size();

        for (size_t i = 0; i < n; ++i) {
            size_t f32_off = snap.f32_offsets[i], f32_end = snap.f32_offsets[i + 1];
            size_t f16_off = snap.f16_offsets[i], f16_end = snap.f16_offsets[i + 1];
            size_t i64_off = snap.i64_offsets[i], i64_end = snap.i64_offsets[i + 1];
            size_t b8_off  = snap.b8_offsets[i],  b8_end  = snap.b8_offsets[i + 1];

            bool has_f32 = f32_end > f32_off;
            bool has_f16 = f16_end > f16_off;
            bool sliced = (snap.shapes[i] != init_shapes[i]) && (has_f32 || has_f16);

            if (sliced) {
                size_t full_size = 1;
                for (auto d : init_shapes[i]) full_size *= (d > 0 ? d : 1);

                int sd = -1;
                for (size_t d = 0; d < init_shapes[i].size(); ++d) {
                    if (snap.shapes[i][d] != init_shapes[i][d]) { sd = (int)d; break; }
                }

                if (has_f16) {
                    f16[0][i].resize(full_size);
                    if (sd >= 0) {
                        int64_t N = snap.shapes[i][sd];
                        int64_t outer = 1;
                        for (int d = 0; d < sd; ++d) outer *= init_shapes[i][d];
                        int64_t inner = 1;
                        for (size_t d = sd + 1; d < init_shapes[i].size(); ++d) inner *= init_shapes[i][d];
                        int64_t full_stride = init_shapes[i][sd] * inner;
                        int64_t slice_stride = N * inner;
                        const uint16_t* src = snap.f16_data.data() + f16_off;
                        uint16_t* dst = f16[0][i].data();
                        for (int64_t o = 0; o < outer; ++o)
                            memcpy(dst + o * full_stride, src + o * slice_stride, slice_stride * sizeof(uint16_t));
                    }
                } else {
                    f32[b][i].resize(full_size);
                    if (sd >= 0) {
                        int64_t N = snap.shapes[i][sd];
                        int64_t outer = 1;
                        for (int d = 0; d < sd; ++d) outer *= init_shapes[i][d];
                        int64_t inner = 1;
                        for (size_t d = sd + 1; d < init_shapes[i].size(); ++d) inner *= init_shapes[i][d];
                        int64_t full_stride = init_shapes[i][sd] * inner;
                        int64_t slice_stride = N * inner;
                        const float* src = snap.f32_data.data() + f32_off;
                        float* dst = f32[b][i].data();
                        for (int64_t o = 0; o < outer; ++o)
                            memcpy(dst + o * full_stride, src + o * slice_stride, slice_stride * sizeof(float));
                    }
                }
            } else {
                f32[b][i].assign(snap.f32_data.begin() + f32_off, snap.f32_data.begin() + f32_end);
                f16[0][i].assign(snap.f16_data.begin() + f16_off, snap.f16_data.begin() + f16_end);
            }

            i64[b][i].assign(snap.i64_data.begin() + i64_off, snap.i64_data.begin() + i64_end);
            b8[b][i].assign(snap.b8_data.begin() + b8_off, snap.b8_data.begin() + b8_end);
        }

        shapes = init_shapes;
    }

    DiskSnapshot snapshot_to_disk(const Snapshot& snap) const {
        DiskSnapshot ds;
        size_t n = names.size();

        size_t total = 8;
        for (size_t i = 0; i < n; ++i) {
            size_t f32_count = snap.f32_offsets[i + 1] - snap.f32_offsets[i];
            size_t f16_count = snap.f16_offsets[i + 1] - snap.f16_offsets[i];
            size_t i64_count = snap.i64_offsets[i + 1] - snap.i64_offsets[i];
            size_t b8_count = snap.b8_offsets[i + 1] - snap.b8_offsets[i];
            total += 4 + snap.shapes[i].size() * 8 + 4 + 8;
            if (types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) total += i64_count * 8;
            else if (types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) total += b8_count;
            else if (types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) total += f16_count * 2;
            else total += f32_count * 4;
        }

        ds.blob.resize(total);
        uint8_t* p = ds.blob.data();
        auto write = [&](const void* src, size_t bytes) { memcpy(p, src, bytes); p += bytes; };

        int32_t cb = snap.current_buf, ns = int32_t(n);
        write(&cb, 4); write(&ns, 4);

        for (size_t i = 0; i < n; ++i) {
            int32_t ndims = int32_t(snap.shapes[i].size());
            int32_t type = int32_t(types[i]);
            write(&ndims, 4);
            write(snap.shapes[i].data(), ndims * 8);
            write(&type, 4);

            size_t f32_count = snap.f32_offsets[i + 1] - snap.f32_offsets[i];
            size_t f16_count = snap.f16_offsets[i + 1] - snap.f16_offsets[i];
            size_t i64_count = snap.i64_offsets[i + 1] - snap.i64_offsets[i];
            size_t b8_count = snap.b8_offsets[i + 1] - snap.b8_offsets[i];

            int64_t data_bytes;
            if (types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                data_bytes = i64_count * 8; write(&data_bytes, 8);
                write(snap.i64_data.data() + snap.i64_offsets[i], data_bytes);
            } else if (types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
                data_bytes = b8_count; write(&data_bytes, 8);
                write(snap.b8_data.data() + snap.b8_offsets[i], data_bytes);
            } else if (types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                data_bytes = f16_count * 2; write(&data_bytes, 8);
                write(snap.f16_data.data() + snap.f16_offsets[i], data_bytes);
            } else {
                data_bytes = f32_count * 4; write(&data_bytes, 8);
                write(snap.f32_data.data() + snap.f32_offsets[i], data_bytes);
            }
        }
        return ds;
    }

    void restore_from_disk(const DiskSnapshot& ds) {
        const uint8_t* p = ds.blob.data();
        auto read = [&](void* dst, size_t bytes) { memcpy(dst, p, bytes); p += bytes; };

        int32_t cb, ns;
        read(&cb, 4); read(&ns, 4);
        current_buf = cb;
        int b = in_buf();

        for (int32_t i = 0; i < ns; ++i) {
            int32_t ndims, type;
            int64_t data_bytes;
            read(&ndims, 4);
            std::vector<int64_t> loaded_shape(ndims);
            read(loaded_shape.data(), ndims * 8);
            read(&type, 4);
            read(&data_bytes, 8);

            if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                size_t count = data_bytes / 8;
                auto* src = reinterpret_cast<const int64_t*>(p);
                i64[b][i].assign(src, src + count);
                p += data_bytes;
            } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
                b8[b][i].assign(p, p + data_bytes);
                p += data_bytes;
            } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                bool sliced = !init_shapes.empty() && loaded_shape != init_shapes[i];
                if (sliced) {
                    size_t full_size = 1;
                    for (auto d : init_shapes[i]) full_size *= (d > 0 ? d : 1);
                    f16[0][i].resize(full_size);
                    int sd = -1;
                    for (size_t d = 0; d < init_shapes[i].size(); ++d) {
                        if (loaded_shape[d] != init_shapes[i][d]) { sd = (int)d; break; }
                    }
                    if (sd >= 0) {
                        int64_t N = loaded_shape[sd];
                        int64_t outer = 1;
                        for (int d = 0; d < sd; ++d) outer *= init_shapes[i][d];
                        int64_t inner = 1;
                        for (size_t d = sd + 1; d < init_shapes[i].size(); ++d) inner *= init_shapes[i][d];
                        int64_t full_stride = init_shapes[i][sd] * inner;
                        int64_t slice_stride = N * inner;
                        const uint16_t* src = reinterpret_cast<const uint16_t*>(p);
                        uint16_t* dst = f16[0][i].data();
                        for (int64_t o = 0; o < outer; ++o)
                            memcpy(dst + o * full_stride, src + o * slice_stride, slice_stride * sizeof(uint16_t));
                    }
                    p += data_bytes;
                } else {
                    size_t count = data_bytes / 2;
                    auto* src = reinterpret_cast<const uint16_t*>(p);
                    f16[0][i].assign(src, src + count);
                    p += data_bytes;
                }
            } else {
                bool sliced = !init_shapes.empty() && loaded_shape != init_shapes[i];
                if (sliced) {
                    size_t full_size = 1;
                    for (auto d : init_shapes[i]) full_size *= (d > 0 ? d : 1);
                    f32[b][i].resize(full_size);
                    int sd = -1;
                    for (size_t d = 0; d < init_shapes[i].size(); ++d) {
                        if (loaded_shape[d] != init_shapes[i][d]) { sd = (int)d; break; }
                    }
                    if (sd >= 0) {
                        int64_t N = loaded_shape[sd];
                        int64_t outer = 1;
                        for (int d = 0; d < sd; ++d) outer *= init_shapes[i][d];
                        int64_t inner = 1;
                        for (size_t d = sd + 1; d < init_shapes[i].size(); ++d) inner *= init_shapes[i][d];
                        int64_t full_stride = init_shapes[i][sd] * inner;
                        int64_t slice_stride = N * inner;
                        const float* src = reinterpret_cast<const float*>(p);
                        float* dst = f32[b][i].data();
                        for (int64_t o = 0; o < outer; ++o)
                            memcpy(dst + o * full_stride, src + o * slice_stride, slice_stride * sizeof(float));
                    }
                    p += data_bytes;
                } else {
                    size_t count = data_bytes / 4;
                    auto* src = reinterpret_cast<const float*>(p);
                    f32[b][i].assign(src, src + count);
                    p += data_bytes;
                }
            }
        }

        shapes = init_shapes;
    }
};

// ── StatefulRunner ──────────────────────────────────────────────────────────
// Combines an OrtSession with a StateBufferIO and an IoBinding to run the
// autoregressive model efficiently. Non-state inputs are passed in per-step;
// state inputs/outputs are managed automatically via double-buffering.

class StatefulRunner {
    OrtSession& sess_;
    Ort::MemoryInfo mem_;
    StateBufferIO state_;
    std::unique_ptr<Ort::IoBinding> binding_;

public:
    StatefulRunner(OrtSession& sess)
        : sess_(sess), mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        state_.init(sess);
        binding_ = std::make_unique<Ort::IoBinding>(sess_.session());
    }

    StateBufferIO& state() { return state_; }

    using Snapshot = StateBufferIO::Snapshot;
    using DiskSnapshot = StateBufferIO::DiskSnapshot;
    Snapshot take_snapshot() const { return state_.take_snapshot(); }
    void restore_snapshot(const Snapshot& snap) { state_.restore_snapshot(snap); }
    void restore_from_disk(const DiskSnapshot& ds) { state_.restore_from_disk(ds); }
    DiskSnapshot snapshot_to_disk(const Snapshot& snap) const { return state_.snapshot_to_disk(snap); }

    // Full re-initialization — creates fresh StateBufferIO from session metadata.
    // Required for models with dynamic states (e.g. main transformer KV cache)
    // where shapes change between runs.
    void reinit() {
        state_ = StateBufferIO();
        state_.init(sess_);
    }

    // Lightweight reset — zeroes existing buffers without reallocation.
    // Only safe for models with fixed-size states (e.g. Mimi decoder).
    void reset_state() {
        state_.reset();
    }

    std::vector<Ort::Value> run(const std::vector<Ort::Value>& non_state_inputs) {
        binding_->ClearBoundInputs();
        binding_->ClearBoundOutputs();

        const auto& in_names = sess_.input_names();
        const auto& out_names = sess_.output_names();

        size_t non_state_idx = 0;
        size_t state_idx = 0;
        for (size_t i = 0; i < in_names.size(); ++i) {
            if (in_names[i].find("state_") == 0) {
                binding_->BindInput(in_names[i].c_str(), state_.create_input_value(state_idx++, mem_));
            } else {
                binding_->BindInput(in_names[i].c_str(), non_state_inputs[non_state_idx++]);
            }
        }

        std::vector<std::pair<size_t, size_t>> dynamic_out_states;
        state_idx = 0;
        for (size_t i = 0; i < out_names.size(); ++i) {
            if (out_names[i].find("out_state_") == 0) {
                if (state_.is_dynamic[state_idx]) {
                    binding_->BindOutput(out_names[i].c_str(), mem_);
                    dynamic_out_states.push_back({i, state_idx});
                } else {
                    binding_->BindOutput(out_names[i].c_str(), state_.create_output_value(state_idx, mem_));
                }
                state_idx++;
            } else {
                binding_->BindOutput(out_names[i].c_str(), mem_);
            }
        }

        sess_.run_with_binding(*binding_);
        auto outputs = binding_->GetOutputValues();
        for (auto& [out_idx, st_idx] : dynamic_out_states) {
            state_.copy_from_output(st_idx, outputs[out_idx]);
        }
        state_.swap();

        std::vector<Ort::Value> result;
        for (size_t i = 0; i < out_names.size(); ++i) {
            if (out_names[i].find("out_state_") != 0) {
                result.push_back(std::move(outputs[i]));
            }
        }
        return result;
    }

    Ort::MemoryInfo& mem() { return mem_; }
};

// ════════════════════════════════════════════════════════════════════════════
// Tokenizer
// ════════════════════════════════════════════════════════════════════════════

class Tokenizer {
    sentencepiece::SentencePieceProcessor proc_;
public:
    explicit Tokenizer(const std::string& path) {
        auto s = proc_.Load(path);
        if (!s.ok()) throw std::runtime_error("Failed to load tokenizer: " + s.ToString());
    }
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        proc_.Encode(text, &ids);
        return ids;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// TTS Engine
// ════════════════════════════════════════════════════════════════════════════

class PocketTTS {
public:
    static constexpr int SR = 24000;

    explicit PocketTTS(const Config& cfg = {}) : cfg_(cfg) {
        rng::seed(uint64_t(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
        // rng::seed(uint64_t(28380294765));
        tok_ = std::make_unique<Tokenizer>(cfg_.tokenizer_path);

        // Thread budget: --threads sets the total. During pipelined streaming,
        // the AR generator and Mimi decoder run simultaneously, so we split the
        // budget between them. Non-pipelined models (encoder, text conditioner)
        // get the full budget since they run alone.
        int cores = std::max(1, int(std::thread::hardware_concurrency()));
        int total = cfg_.num_threads ? cfg_.num_threads : std::max(2, cores / 2);
        total = std::min(total, 8);  // cap at 8
        int threads_full = total;
        // Ensure that there are 3x decoder threads as AR threads
        // this seems to prevent robotic voice in output
        total = std::max(total, 3);
        int threads_ar = total / 3;
        int threads_dec = total - threads_ar;

        auto make_opts = [](int threads) {
            Ort::SessionOptions opts;
            opts.SetIntraOpNumThreads(threads);
            opts.SetInterOpNumThreads(1);
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            return opts;
        };

        // Arena disabled for sessions with large or variable-size inputs that
        // run infrequently — ORT's arena never releases memory back to the OS,
        // so a single large allocation permanently inflates RSS.
        //   - mimi_encoder: processes up to 720k float samples on cache miss
        //   - mimi_decoder: reset per sentence during streaming
        auto make_opts_no_arena = [](int threads) {
            Ort::SessionOptions opts;
            opts.SetIntraOpNumThreads(threads);
            opts.SetInterOpNumThreads(1);
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            #ifndef __ANDROID__
            opts.DisableMemPattern();
            #endif
            opts.DisableCpuMemArena();
            return opts;
        };

        auto opts_full = make_opts(threads_full);
        auto opts_enc = make_opts_no_arena(threads_full);
        auto opts_ar = make_opts(threads_ar);
        auto opts_dec = make_opts_no_arena(threads_dec);

        if (cfg_.verbose) {
            std::cout << "\n========== ONNX RUNTIME INFO ==========\n";
            std::cout << "  ORT Version: " << OrtGetApiBase()->GetVersionString() << "\n";
            std::cout << "  Thread budget: AR: " << threads_ar
                      << ", decoder: " << threads_dec << ", full: " << threads_full << "\n";
            auto providers = Ort::GetAvailableProviders();
            std::cout << "  Execution Providers: ";
            for (size_t i = 0; i < providers.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << providers[i];
            }
            std::cout << "\n================================\n";
        }

        auto& env = get_ort_env();
        std::string sfx = cfg_.precision == "int8" ? "_int8" : "";

        enc_ = std::make_unique<OrtSession>(env, cfg_.models_dir + "/mimi_encoder.onnx", opts_enc, "mimi_encoder");
        txt_ = std::make_unique<OrtSession>(env, cfg_.models_dir + "/text_conditioner.onnx", opts_full, "text_conditioner");
        main_ = std::make_unique<OrtSession>(env, cfg_.models_dir + "/flow_lm_main" + sfx + ".onnx", opts_ar, "flow_lm_main" + sfx);
        flow_ = std::make_unique<OrtSession>(env, cfg_.models_dir + "/flow_lm_flow" + sfx + ".onnx", opts_ar, "flow_lm_flow" + sfx);
        dec_ = std::make_unique<OrtSession>(env, cfg_.models_dir + "/mimi_decoder" + sfx + ".onnx", opts_dec, "mimi_decoder" + sfx);

        main_runner_ = std::make_unique<StatefulRunner>(*main_);
        dec_runner_ = std::make_unique<StatefulRunner>(*dec_);

        dt_ = 1.0f / cfg_.lsd_steps;
        st_values_.reserve(cfg_.lsd_steps);
        for (int j = 0; j < cfg_.lsd_steps; ++j) {
            float s = float(j) / cfg_.lsd_steps;
            st_values_.emplace_back(s, s + dt_);
        }

        if (cfg_.verbose) {
            std::cerr << "\n========== MODEL INFO ==========\n";
            main_->print_info();
            dec_->print_info();
            std::cerr << "================================\n";
        }
    }

    // ── Audio I/O ───────────────────────────────────────────────────────────

    static AudioData load_audio(const std::string& path) {
        auto _ = g_prof.time("load_audio");

        float* raw = nullptr;
        unsigned ch = 0, sr = 0;
        drwav_uint64 n = 0;

        raw = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &ch, &sr, &n, nullptr);

        if (!raw) throw std::runtime_error("Failed to load audio: " + path);

        std::vector<float> mono(n);
        for (size_t i = 0; i < n; ++i) {
            float sum = 0;
            for (unsigned c = 0; c < ch; ++c) sum += raw[i * ch + c];
            mono[i] = sum / ch;
        }

        drwav_free(raw, nullptr);

        if (sr != SR) {
            auto _ = g_prof.time("resample");
            mono = resample(mono, sr, SR);
        }

        float mx = *std::max_element(mono.begin(), mono.end(), [](float a, float b) { return std::abs(a) < std::abs(b); });
        if (std::abs(mx) > 1) for (auto& s : mono) s /= std::abs(mx);

        return {std::move(mono), SR};
    }

    static void save_audio(const AudioData& a, const std::string& path) {
        auto _ = g_prof.time("save_audio");
        drwav w;
        drwav_data_format fmt{drwav_container_riff, DR_WAVE_FORMAT_IEEE_FLOAT, 1, drwav_uint32(a.sample_rate), 32};
        if (!drwav_init_file_write(&w, path.c_str(), &fmt, nullptr))
            throw std::runtime_error("Failed to write: " + path);
        drwav_write_pcm_frames(&w, a.samples.size(), a.samples.data());
        drwav_uninit(&w);
    }

    // ── Voice Encoding ──────────────────────────────────────────────────────

    Tensor encode_voice(const std::string& path) {
        auto timer = g_prof.time("encode_voice");

        if (cfg_.voice_cache) {
            std::string cache_path = cache::get_cache_path(cfg_.voices_dir, path);

            if (cache::is_cache_valid(path, cache_path)) {
                auto cache_timer = g_prof.time("encode_voice.cache_load");
                std::vector<int64_t> shape;
                std::vector<float> data;
                if (cache::load_embedding(cache_path, shape, data)) {
                    if (cfg_.verbose) {
                        std::cerr << "  Loaded cached embedding: " << cache_path << "\n";
                    }
                    return Tensor(std::move(data), std::move(shape));
                }
            }
        }

        auto a = load_audio(path);

        // Truncate to 30 seconds max — matches Python, prevents OOM on long samples
        static constexpr size_t MAX_VOICE_SAMPLES = 30 * SR;  // 720000 at 24kHz
        if (a.samples.size() > MAX_VOICE_SAMPLES) {
            a.samples.resize(MAX_VOICE_SAMPLES);
            if (cfg_.verbose) std::cerr << "  Voice truncated to 30s\n";
        }

        Tensor t({1, 1, int64_t(a.samples.size())});
        std::copy(a.samples.begin(), a.samples.end(), t.data.begin());

        Ort::MemoryInfo m = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> in;
        in.push_back(Ort::Value::CreateTensor<float>(m, t.ptr(), t.numel(), t.shape.data(), t.shape.size()));

        auto out = enc_->run(in);
        auto sh = out[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t n = 1;
        for (auto d : sh) n *= d;

        Tensor r(std::vector<float>(out[0].GetTensorData<float>(), out[0].GetTensorData<float>() + n),
                 std::vector<int64_t>(sh.begin(), sh.end()));

        while (r.shape.size() > 3) r = r.squeeze(0);
        if (r.shape.size() < 3) r.reshape({1, r.shape[0], r.shape[1]});

        if (cfg_.voice_cache) {
            std::string cache_path = cache::get_cache_path(cfg_.voices_dir, path);
            if (cache::save_embedding(cache_path, r.shape, r.data)) {
                if (cfg_.verbose) {
                    std::cerr << "  Saved embedding cache: " << cache_path << "\n";
                }
            }
        }

        return r;
    }

    // ── Public API ──────────────────────────────────────────────────────────

    AudioData generate(const std::string& text, const std::string& voice, int max_frames = 500) {
        return generate(text, get_voice(voice), max_frames);
    }

    AudioData generate(const std::string& text, const Tensor& voice, int max_frames = 500);
    void stream(const std::string& text, const std::string& voice, StreamCallback cb, int max_frames = 500);
    void stream(const std::string& text, const Tensor& voice, StreamCallback cb, int max_frames = 500);
    const Config& config() const { return cfg_; }

    void prepare_voice(const std::string& voice) {
        std::string p = voice;
        if (p.size() <= 4 || p.substr(p.size() - 4) != ".wav") p += ".wav";
        voice_kv_path_ = p;
        std::string resolved = resolve_voice_path(p);
        encode_voice(resolved);
    }

    void clear_vcache() {
        vcache_.clear();
    }

    double warmup() {
        auto start = std::chrono::high_resolution_clock::now();
        Tensor dummy_voice({1, 8, 1024});
        std::fill(dummy_voice.data.begin(), dummy_voice.data.end(), 0.0f);
        stream("Hi.", dummy_voice, [](const float*, size_t) { return true; }, 1);
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    void print_profiling_report() const { g_prof.report(); }
    void reset_profiling() { g_prof.reset(); }

private:
    Config cfg_;
    std::unique_ptr<OrtSession> enc_, txt_, main_, flow_, dec_;
    std::unique_ptr<Tokenizer> tok_;
    std::unique_ptr<StatefulRunner> main_runner_;
    std::unique_ptr<StatefulRunner> dec_runner_;  // reused across stream() calls
    std::vector<std::pair<float, float>> st_values_;
    float dt_;
    std::unordered_map<std::string, Tensor> vcache_;

    // ── Voice Resolution ────────────────────────────────────────────────────

    std::string resolve_voice_path(const std::string& p) const {
        if (!p.empty() && p[0] == '/') return p;
        return cfg_.voices_dir + "/" + p;
    }

    const Tensor& get_voice(const std::string& voice) {
        std::string p = voice;
        if (p.size() <= 4 || p.substr(p.size() - 4) != ".wav") p += ".wav";
        voice_kv_path_ = p;
        auto it = vcache_.find(p);
        if (it != vcache_.end()) return it->second;

        std::string resolved = resolve_voice_path(p);
        return vcache_[p] = encode_voice(resolved);
    }

    // ── Tokenization ────────────────────────────────────────────────────────

    TensorI64 tokenize(const std::string& t) {
        auto _ = g_prof.time("tokenize");
        auto ids = tok_->encode(t);
        if (cfg_.verbose) std::cerr << "  Tokens: " << ids.size() << " from " << t.size() << " chars\n";
        TensorI64 r({1, int64_t(ids.size())});
        for (size_t i = 0; i < ids.size(); ++i) r.data[i] = ids[i];
        return r;
    }

    // ── LatentGen ───────────────────────────────────────────────────────────
    // Autoregressive latent generator. Each call to next() runs the main
    // transformer for one frame, then solves the flow matching ODE to produce
    // a 32-dim latent vector for the Mimi decoder.
    //
    // Construction has two paths:
    //   - Full: runs voice conditioning + text conditioning from scratch
    //   - Cached: restores a KV snapshot then runs text conditioning only

    class LatentGen {
        PocketTTS& tts;
        int max_, idx_ = 0, extra_ = 0;
        int eos_frame_ = -1;
        int eos_extra_;  // frames to generate after EOS
        bool done_ = false, eos_ = false;
        float temp_;
        Ort::MemoryInfo m_;

        StatefulRunner& main_runner_;

        std::vector<float> fx_, cl_, cond_, temb_;
        std::vector<int64_t> csh_, tsh_;
        std::vector<Ort::Value> flow_inputs_;

        static constexpr int64_t curr_shape_[3] = {1, 1, 32};
        static constexpr int64_t empty_text_shape_[3] = {1, 0, 1024};
        static constexpr int64_t empty_seq_shape_[3] = {1, 0, 32};
        static constexpr int64_t s_shape_[2] = {1, 1};
        static constexpr int64_t x_shape_[2] = {1, 32};

        std::vector<float> s_buf_{1}, t_buf_{1};

        void cond_pass(const float* d, size_t sz, const std::vector<int64_t>& sh) {
            std::vector<Ort::Value> inputs;
            inputs.push_back(Ort::Value::CreateTensor<float>(m_, nullptr, 0, empty_seq_shape_, 3));
            inputs.push_back(Ort::Value::CreateTensor<float>(m_, const_cast<float*>(d), sz, sh.data(), sh.size()));
            main_runner_.run(inputs);
        }

    public:
        using Snapshot = StatefulRunner::Snapshot;

        // Full path: voice conditioning → (optional snapshot) → text conditioning
        LatentGen(PocketTTS& t, const Tensor& v, const TensorI64& tid, int max, int eos_extra, Snapshot* out_voice_snap = nullptr)
            : tts(t), max_(max), eos_extra_(eos_extra), m_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
              main_runner_(*tts.main_runner_),
              fx_(32, 0), cl_(32, std::numeric_limits<float>::quiet_NaN()) {
            temp_ = std::sqrt(tts.cfg_.temperature);
            flow_inputs_.reserve(4);

            main_runner_.reinit();

            {
                auto _ = g_prof.time("text_conditioning");
                std::vector<Ort::Value> in;
                in.push_back(Ort::Value::CreateTensor<int64_t>(m_, const_cast<int64_t*>(tid.ptr()), tid.numel(), tid.shape.data(), tid.shape.size()));
                auto out = tts.txt_->run(in);

                auto sh = out[0].GetTensorTypeAndShapeInfo().GetShape();
                size_t n = 1;
                for (auto d : sh) n *= d;
                temb_.assign(out[0].GetTensorData<float>(), out[0].GetTensorData<float>() + n);
                tsh_.assign(sh.begin(), sh.end());
                if (tsh_.size() == 2) tsh_.insert(tsh_.begin(), 1);
            }

            {
                auto _ = g_prof.time("voice_conditioning_pass");
                cond_pass(v.ptr(), v.numel(), v.shape);
            }

            if (out_voice_snap) *out_voice_snap = main_runner_.take_snapshot();

            {
                auto _ = g_prof.time("text_conditioning_pass");
                cond_pass(temb_.data(), temb_.size(), tsh_);
            }
        }

        // Cached path: restore KV snapshot → text conditioning only
        LatentGen(PocketTTS& t, const Snapshot& voice_snap, const TensorI64& tid, int max, int eos_extra)
            : tts(t), max_(max), eos_extra_(eos_extra), m_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
              main_runner_(*tts.main_runner_),
              fx_(32, 0), cl_(32, std::numeric_limits<float>::quiet_NaN()) {
            temp_ = std::sqrt(tts.cfg_.temperature);
            flow_inputs_.reserve(4);

            {
                auto _ = g_prof.time("text_conditioning");
                std::vector<Ort::Value> in;
                in.push_back(Ort::Value::CreateTensor<int64_t>(m_, const_cast<int64_t*>(tid.ptr()), tid.numel(), tid.shape.data(), tid.shape.size()));
                auto out = tts.txt_->run(in);

                auto sh = out[0].GetTensorTypeAndShapeInfo().GetShape();
                size_t n = 1;
                for (auto d : sh) n *= d;
                temb_.assign(out[0].GetTensorData<float>(), out[0].GetTensorData<float>() + n);
                tsh_.assign(sh.begin(), sh.end());
                if (tsh_.size() == 2) tsh_.insert(tsh_.begin(), 1);
            }

            {
                auto _ = g_prof.time("voice_kv_restore");
                main_runner_.restore_snapshot(voice_snap);
            }

            {
                auto _ = g_prof.time("text_conditioning_pass");
                cond_pass(temb_.data(), temb_.size(), tsh_);
            }
        }

        bool has_next() const { return !done_ && idx_ < max_; }

        Tensor next() {
            if (!has_next()) throw std::runtime_error("No more latents");

            float eos_logit = 0;

            {
                auto _ = g_prof.time("frame:main_model");
                std::vector<Ort::Value> inputs;
                inputs.push_back(Ort::Value::CreateTensor<float>(m_, cl_.data(), cl_.size(), curr_shape_, 3));
                inputs.push_back(Ort::Value::CreateTensor<float>(m_, nullptr, 0, empty_text_shape_, 3));

                auto outputs = main_runner_.run(inputs);

                auto csh = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
                size_t cn = 1;
                for (auto d : csh) cn *= d;
                cond_.assign(outputs[0].GetTensorData<float>(), outputs[0].GetTensorData<float>() + cn);
                csh_.assign(csh.begin(), csh.end());

                eos_logit = outputs[1].GetTensorData<float>()[0];
            }

            if (!eos_ && eos_logit > tts.cfg_.eos_threshold) {
                eos_ = true;
                eos_frame_ = idx_;
            }

            if (eos_) {
                if (++extra_ > eos_extra_) {
                    done_ = true;
                    return Tensor();
                }
            }

            {
                auto _ = g_prof.time("frame:rng");
                if (temp_ > 0) {
                    rng::fill_normal(fx_.data(), 32, 0, temp_);
                    float nc = tts.cfg_.noise_clamp;
                    if (nc > 0) for (auto& v : fx_) v = std::max(-nc, std::min(nc, v));
                }
                else std::fill(fx_.begin(), fx_.end(), 0.0f);
            }

            {
                auto _ = g_prof.time("frame:flow_steps");
                for (const auto& [s, t] : tts.st_values_) {
                    s_buf_[0] = s;
                    t_buf_[0] = t;

                    flow_inputs_.clear();
                    flow_inputs_.push_back(Ort::Value::CreateTensor<float>(m_, cond_.data(), cond_.size(), csh_.data(), csh_.size()));
                    flow_inputs_.push_back(Ort::Value::CreateTensor<float>(m_, s_buf_.data(), 1, s_shape_, 2));
                    flow_inputs_.push_back(Ort::Value::CreateTensor<float>(m_, t_buf_.data(), 1, s_shape_, 2));
                    flow_inputs_.push_back(Ort::Value::CreateTensor<float>(m_, fx_.data(), 32, x_shape_, 2));

                    auto fo = tts.flow_->run(flow_inputs_);
                    const float* out_data = fo[0].GetTensorData<float>();

                    for (int i = 0; i < 32; ++i)
                        fx_[i] += out_data[i] * tts.dt_;
                }
            }

            std::copy(fx_.begin(), fx_.end(), cl_.begin());
            idx_++;
            return Tensor({fx_.begin(), fx_.end()}, {1, 1, 32});
        }

        int frame_idx() const { return idx_; }
        int eos_frame() const { return eos_frame_; }
    };

    friend class LatentGen;

    // ── Voice KV Cache ──────────────────────────────────────────────────────
    // Three-tier cache for voice-conditioned KV state:
    //   1. In-memory snapshot (fastest, ~1ms restore)
    //   2. On-disk .kv file (fast, ~4ms restore)
    //   3. Full recomputation (slow, hundreds of ms)

    using VoiceKVSnapshot = LatentGen::Snapshot;

    std::unique_ptr<VoiceKVSnapshot> voice_kv_snap_;
    uint64_t voice_kv_hash_ = 0;
    std::string voice_kv_path_;

    static uint64_t voice_hash(const Tensor& v) {
        uint64_t h = 14695981039346656037ull;
        int n = std::min(16, int(v.data.size()));
        for (int i = 0; i < n; ++i) {
            uint32_t bits;
            memcpy(&bits, &v.data[i], sizeof(bits));
            h ^= bits;
            h *= 1099511628211ull;
        }
        return h;
    }

    LatentGen make_gen(const Tensor& v, const TensorI64& t, int max, int eos_extra) {
        uint64_t vh = voice_hash(v);

        // Tier 1: in-memory cache hit
        if (voice_kv_snap_ && voice_kv_hash_ == vh) {
            return LatentGen(*this, *voice_kv_snap_, t, max, eos_extra);
        }

        // Tier 2: disk cache hit
        if (cfg_.voice_cache && !voice_kv_path_.empty()) {
            std::string kv_path = cache::get_cache_path(cfg_.voices_dir, voice_kv_path_, "kv");
            std::string resolved = resolve_voice_path(voice_kv_path_);
            StateBufferIO::DiskSnapshot ds;
            if (cache::is_cache_valid(resolved, kv_path) && ds.load_from_disk(kv_path)) {
                if (cfg_.verbose) std::cerr << "  Loaded KV cache: " << kv_path << "\n";
                main_runner_->restore_from_disk(ds);
                voice_kv_snap_ = std::make_unique<VoiceKVSnapshot>(main_runner_->take_snapshot());
                voice_kv_hash_ = vh;
                return LatentGen(*this, *voice_kv_snap_, t, max, eos_extra);
            }
        }

        // Tier 3: full voice conditioning
        VoiceKVSnapshot snap;
        auto gen = LatentGen(*this, v, t, max, eos_extra, &snap);
        voice_kv_snap_ = std::make_unique<VoiceKVSnapshot>(std::move(snap));
        voice_kv_hash_ = vh;

        if (cfg_.voice_cache && !voice_kv_path_.empty()) {
            std::string kv_path = cache::get_cache_path(cfg_.voices_dir, voice_kv_path_, "kv");
            auto ds = main_runner_->snapshot_to_disk(*voice_kv_snap_);
            if (ds.save_to_disk(kv_path)) {
                if (cfg_.verbose) std::cerr << "  Saved KV cache: " << kv_path << "\n";
            }
        }

        return gen;
    }
};

// Required for C++17 ODR-use of constexpr static members
constexpr int64_t PocketTTS::LatentGen::curr_shape_[3];
constexpr int64_t PocketTTS::LatentGen::empty_text_shape_[3];
constexpr int64_t PocketTTS::LatentGen::empty_seq_shape_[3];
constexpr int64_t PocketTTS::LatentGen::s_shape_[2];
constexpr int64_t PocketTTS::LatentGen::x_shape_[2];

// ── Out-of-line method definitions ──────────────────────────────────────────

AudioData PocketTTS::generate(const std::string& text, const Tensor& voice, int max_frames) {
    auto _ = g_prof.time("generate_total");

    auto sentences = split_sentences(text);

    if (sentences.size() == 1) {
        std::vector<float> samples;
        samples.reserve(max_frames * 2000);
        stream(sentences[0], voice, [&](const float* s, size_t n) {
            samples.insert(samples.end(), s, s + n);
            return true;
        }, max_frames);
        return {std::move(samples), SR};
    }

    // Multi-sentence: generate each independently, crossfade at boundaries
    static constexpr int XFADE_SAMPLES = 240;  // 10ms at 24kHz

    std::vector<float> all_samples;

    for (size_t i = 0; i < sentences.size(); ++i) {
        if (cfg_.verbose) {
            std::cerr << "  Sentence " << (i + 1) << "/" << sentences.size()
                      << ": \"" << sentences[i].substr(0, 60)
                      << (sentences[i].size() > 60 ? "..." : "") << "\"\n";
        }

        std::vector<float> chunk_samples;
        stream(sentences[i], voice, [&](const float* s, size_t n) {
            chunk_samples.insert(chunk_samples.end(), s, s + n);
            return true;
        }, max_frames);

        if (chunk_samples.empty()) continue;

        if (i > 0 && !all_samples.empty()) {
            int xfade = std::min(XFADE_SAMPLES, std::min(int(all_samples.size()), int(chunk_samples.size())));
            size_t tail_start = all_samples.size() - xfade;
            for (int j = 0; j < xfade; ++j) {
                float t = float(j) / float(xfade);
                all_samples[tail_start + j] = all_samples[tail_start + j] * (1.0f - t) + chunk_samples[j] * t;
            }
            all_samples.insert(all_samples.end(), chunk_samples.begin() + xfade, chunk_samples.end());
        } else {
            all_samples.insert(all_samples.end(), chunk_samples.begin(), chunk_samples.end());
        }
    }

    return {std::move(all_samples), SR};
}

void PocketTTS::stream(const std::string& text, const std::string& voice, StreamCallback cb, int max_frames) {
    stream(text, get_voice(voice), cb, max_frames);
}

void PocketTTS::stream(const std::string& text, const Tensor& voice, StreamCallback cb, int max_frames) {
    auto sentences = split_sentences(text);

    for (size_t si = 0; si < sentences.size(); ++si) {
        auto& prepared = sentences[si];
        if (prepared.empty()) continue;
        auto eos_extra = calc_eos_extra(prepared, cfg_.eos_extra_frames);
        auto gen = make_gen(voice, tokenize(prepared), max_frames, eos_extra);
        dec_runner_->reset_state();  // zero existing buffers, no reallocation

        // Pipelined: generator thread produces latent frames into a queue,
        // decoder (main thread) consumes them in chunks. The two ONNX sessions
        // (flow_lm_main and mimi_decoder) run on separate threads simultaneously.

        std::mutex mtx;
        std::condition_variable cv;
        std::deque<Tensor> queue;
        bool gen_done = false;
        bool aborted = false;

        std::thread gen_thread([&]() {
            while (gen.has_next()) {
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    if (aborted) return;
                }
                auto f = gen.next();
                if (f.numel() == 0) break;
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    if (aborted) return;
                    queue.push_back(std::move(f));
                }
                cv.notify_one();
            }
            {
                std::lock_guard<std::mutex> lock(mtx);
                gen_done = true;
            }
            cv.notify_one();
        });

        bool first = true;

        while (true) {
            int want = first ? cfg_.first_chunk_frames : cfg_.max_chunk_frames;

            std::vector<Tensor> batch;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&]{ return (int)queue.size() >= want || gen_done || aborted; });
                if (aborted) break;

                int take = gen_done ? (int)queue.size() : std::min((int)queue.size(), want);
                for (int i = 0; i < take; ++i) {
                    batch.push_back(std::move(queue.front()));
                    queue.pop_front();
                }
            }

            if (batch.empty() && gen_done) break;

            if (!batch.empty()) {
                auto lat = Tensor::concat(batch, 1);
                std::vector<Ort::Value> inputs;
                inputs.push_back(Ort::Value::CreateTensor<float>(dec_runner_->mem(), lat.ptr(), lat.numel(),
                                                                  lat.shape.data(), lat.shape.size()));

                auto outputs = dec_runner_->run(inputs);
                auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
                size_t n = 1;
                for (auto d : shape) n *= d;

                if (!cb(outputs[0].GetTensorData<float>(), n)) {
                    std::lock_guard<std::mutex> lock(mtx);
                    aborted = true;
                    break;
                }
                first = false;
            }
        }

        if (gen_thread.joinable()) {
            gen_thread.join();
        }
        if (aborted) return;
    }
}


} // namespace pocket_tts

// ════════════════════════════════════════════════════════════════════════════
// C API (FFI)
// ════════════════════════════════════════════════════════════════════════════

extern "C" {

void* ptt_create(const char* models_dir, const char* voices_dir,
                 const char* tokenizer_path, const char* precision,
                 float temperature, int lsd_steps, int num_threads) {
    try {
        pocket_tts::Config cfg;
        if (models_dir) cfg.models_dir = models_dir;
        if (voices_dir) cfg.voices_dir = voices_dir;
        if (tokenizer_path) cfg.tokenizer_path = tokenizer_path;
        if (precision) cfg.precision = precision;
        cfg.temperature = temperature;
        cfg.lsd_steps = lsd_steps;
        cfg.num_threads = num_threads;
        return new pocket_tts::PocketTTS(cfg);
    } catch (const std::exception& e) {
        std::cerr << "[pocket-tts] init error: " << e.what() << "\n";
        return nullptr;
    }
}

double ptt_warmup(void* handle) {
    if (!handle) return -1;
    try {
        return static_cast<pocket_tts::PocketTTS*>(handle)->warmup();
    } catch (const std::exception& e) {
        std::cerr << "[pocket-tts] warmup error: " << e.what() << "\n";
        return -1;
    }
}

void ptt_free_audio(float* samples) {
    free(samples);
}

void ptt_destroy(void* handle) {
    delete static_cast<pocket_tts::PocketTTS*>(handle);
}

// ── Streaming API ───────────────────────────────────────────────────────────

struct ptt_stream_ctx {
    std::thread thread;
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<std::pair<float*, size_t>> chunks;
    bool done = false;
    bool aborted = false;
};

void* ptt_stream_start(void* handle, const char* text, const char* voice) {
    if (!handle || !text || !voice) return nullptr;
    auto* tts = static_cast<pocket_tts::PocketTTS*>(handle);
    auto* ctx = new ptt_stream_ctx();

    ctx->thread = std::thread([tts, t = std::string(text), v = std::string(voice), ctx]() {
        try {
            tts->stream(t, v, [ctx](const float* samples, size_t n) -> bool {
                float* copy = static_cast<float*>(malloc(n * sizeof(float)));
                if (!copy) return false;
                std::memcpy(copy, samples, n * sizeof(float));
                {
                    std::lock_guard<std::mutex> lock(ctx->mtx);
                    if (ctx->aborted) { free(copy); return false; }
                    ctx->chunks.push_back({copy, n});
                }
                ctx->cv.notify_one();
                return true;
            });
        } catch (const std::exception& e) {
            std::cerr << "[pocket-tts] stream error: " << e.what() << "\n";
        }
        {
            std::lock_guard<std::mutex> lock(ctx->mtx);
            ctx->done = true;
        }
        ctx->cv.notify_one();
    });

    return ctx;
}

int ptt_stream_read(void* stream_ctx, float** out_samples, int* out_len) {
    if (!stream_ctx || !out_samples || !out_len) return -1;
    auto* ctx = static_cast<ptt_stream_ctx*>(stream_ctx);

    std::unique_lock<std::mutex> lock(ctx->mtx);
    ctx->cv.wait(lock, [ctx]{ return !ctx->chunks.empty() || ctx->done; });

    if (!ctx->chunks.empty()) {
        auto [ptr, len] = ctx->chunks.front();
        ctx->chunks.pop_front();
        *out_samples = ptr;
        *out_len = static_cast<int>(len);
        return 1;
    }
    return 0;
}

void ptt_stream_end(void* stream_ctx) {
    if (!stream_ctx) return;
    auto* ctx = static_cast<ptt_stream_ctx*>(stream_ctx);
    {
        std::lock_guard<std::mutex> lock(ctx->mtx);
        ctx->aborted = true;
    }
    ctx->cv.notify_all();
    if (ctx->thread.joinable()) ctx->thread.join();
    for (auto& [ptr, len] : ctx->chunks) free(ptr);
    delete ctx;
}

} // extern "C"

