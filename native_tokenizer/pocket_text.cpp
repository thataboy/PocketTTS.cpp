// Native implementation of the Unigram/Metaspace tokenizer used by Pocket TTS.
// Viterbi follows Hugging Face tokenizers 0.22.2's optimized Unigram algorithm
// (Apache-2.0), including double scores, tie ordering and UTF-8 byte fallback.
#include "pocket_text.h"
#include "unicode_data.hpp"
#include "../json.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pocket_text {
using Json = nlohmann::json;
const std::string marker = "\xE2\x96\x81";
const std::string replacement = "\xEF\xBF\xBD";

struct Rune { uint32_t code; size_t size; };
Rune rune(const std::string& s, size_t pos) {
    const auto first = static_cast<unsigned char>(s.at(pos));
    if (first < 0x80) return {first, 1};
    size_t size = first >= 0xC2 && first <= 0xDF ? 2 : first <= 0xEF && first >= 0xE0 ? 3 :
                  first >= 0xF0 && first <= 0xF4 ? 4 : 0;
    if (!size || pos + size > s.size()) throw std::runtime_error("Invalid UTF-8 text");
    uint32_t cp = first & (0x7F >> size);
    for (size_t i = 1; i < size; ++i) {
        auto b = static_cast<unsigned char>(s[pos + i]);
        if ((b & 0xC0) != 0x80) throw std::runtime_error("Invalid UTF-8 text");
        cp = (cp << 6) | (b & 0x3F);
    }
    if ((size == 2 && cp < 0x80) || (size == 3 && cp < 0x800) ||
        (size == 4 && cp < 0x10000) || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
        throw std::runtime_error("Invalid UTF-8 text");
    return {cp, size};
}

template<size_t N> bool in_ranges(uint32_t c, const UnicodeRange (&ranges)[N]) {
    auto it = std::lower_bound(std::begin(ranges), std::end(ranges), c,
        [](const UnicodeRange& range, uint32_t value) { return range.last < value; });
    return it != std::end(ranges) && it->first <= c;
}
bool space(uint32_t c) { return in_ranges(c, spaces); }
bool digit(uint32_t c) { return in_ranges(c, digits); }

struct Character { uint32_t code; size_t start, size; };
std::vector<Character> characters(const std::string& text) {
    std::vector<Character> result;
    for (size_t i = 0; i < text.size();) {
        auto c = rune(text, i);
        result.push_back({c.code, i, c.size}); i += c.size;
    }
    return result;
}
template<class Predicate> std::string trim_end(const std::string& text, Predicate predicate) {
    auto chars = characters(text);
    size_t end = text.size();
    for (auto it = chars.rbegin(); it != chars.rend() && predicate(it->code); ++it) end = it->start;
    return text.substr(0, end);
}
std::string trim(const std::string& text) {
    size_t start = 0;
    while (start < text.size()) { auto c = rune(text, start); if (!space(c.code)) break; start += c.size; }
    return trim_end(text.substr(start), space);
}
std::vector<std::string> words(const std::string& text) {
    std::vector<std::string> result;
    size_t start = 0;
    for (size_t i = 0; i < text.size();) {
        auto c = rune(text, i);
        if (space(c.code)) { if (i > start) result.push_back(text.substr(start, i - start)); start = i + c.size; }
        i += c.size;
    }
    if (start < text.size()) result.push_back(text.substr(start));
    return result;
}
bool terminal(uint32_t c) { return c == '.' || c == '!' || c == '?' || c == 0x2026; }
bool weak(uint32_t c) { return c == ',' || c == ';' || c == ':' || c == '-' || c == 0x2013 || c == 0x2014; }
bool closer(uint32_t c) { return c == '"' || c == '\'' || c == 0x201D || c == 0x2019 || c == ')' || c == ']' || c == 0xBB || c == ' '; }

std::pair<std::string, int> prepare(const std::string& raw, const Json& config) {
    std::string text = trim(raw);
    auto changes = config.find("replace_characters");
    if (changes != config.end() && changes->is_object() && !changes->empty()) {
        std::string translated;
        for (auto c : characters(text)) {
            auto part = text.substr(c.start, c.size);
            auto value = changes->find(part);
            translated += value == changes->end() ? part : value->get<std::string>();
        }
        text.clear();
        for (const auto& word : words(translated)) { if (!text.empty()) text += ' '; text += word; }
        // Equivalent to Python re.sub(r"([.!?…])\s*[,;:]", r"\1", text).
        auto chars = characters(text);
        std::string cleaned;
        for (size_t i = 0; i < chars.size(); ++i) {
            auto c = chars[i]; cleaned += text.substr(c.start, c.size);
            if (terminal(c.code)) {
                size_t j = i + 1;
                while (j < chars.size() && space(chars[j].code)) ++j;
                if (j < chars.size() && (chars[j].code == ',' || chars[j].code == ';' || chars[j].code == ':')) i = j;
            }
        }
        text = std::move(cleaned);
    }
    if (text.empty()) throw std::runtime_error("Text prompt cannot be empty");
    std::replace(text.begin(), text.end(), '\n', ' ');
    std::replace(text.begin(), text.end(), '\r', ' ');
    for (size_t pos = 0; (pos = text.find("  ", pos)) != std::string::npos; ++pos) text.replace(pos, 2, " ");
    if (config.value("remove_semicolons", false)) std::replace(text.begin(), text.end(), ';', ',');
    const size_t count = words(text).size();
    if (config.value("capitalize_first_letter", true)) {
        auto first = rune(text, 0);
        auto it = std::lower_bound(std::begin(uppercase), std::end(uppercase), first.code,
            [](const Uppercase& item, uint32_t cp) { return item.code < cp; });
        if (it != std::end(uppercase) && it->code == first.code) text = it->text + text.substr(first.size);
    }
    if (config.value("append_terminal_punctuation", true)) {
        auto core = trim_end(text, closer);
        auto chars = characters(core);
        if (!chars.empty() && !terminal(chars.back().code)) {
            if (weak(chars.back().code)) {
                auto tail = trim(text.substr(core.size()));
                text = trim_end(core, [](uint32_t c) { return weak(c) || c == ' '; }) + "." + tail;
            } else text += '.';
        }
    }
    if (config.value("pad_with_spaces_for_short_inputs", false) && count < 5) text = "        " + text;
    return {text, count <= 4 ? 3 : 1};
}

class Tokenizer {
    struct Node { std::unordered_map<unsigned char, size_t> next; int id = -1; };
    std::vector<Node> trie_{1};
    std::vector<std::string> vocab_;
    std::vector<double> scores_;
    std::unordered_map<std::string, int> ids_;
    std::vector<std::pair<std::string, int>> added_;
    std::vector<bool> special_;
    std::array<int, 256> bytes_;
    int unk_;
    double min_score_ = std::numeric_limits<double>::infinity();

    static void check_normalizer(const Json& normalizer) {
        if (normalizer.at("type") == "Prepend" && normalizer.at("prepend") == marker) return;
        if (normalizer.at("type") == "Sequence" && normalizer.at("normalizers").size() == 1) {
            check_normalizer(normalizer.at("normalizers")[0]); return;
        }
        throw std::runtime_error("Unsupported JSON tokenizer normalizer");
    }
    void unigram(const std::string& text, std::vector<int>& result) const {
        struct Best { double score = 0; size_t start = std::string::npos; int id = -1; };
        std::vector<Best> best(text.size() + 1);
        for (size_t start = 0; start < text.size();) {
            const auto width = rune(text, start).size;
            bool single = false;
            auto update = [&](size_t end, int id, double score) {
                const double candidate = score + best[start].score;
                if (best[end].start == std::string::npos || candidate > best[end].score)
                    best[end] = {candidate, start, id};
            };
            size_t node = 0;
            for (size_t end = start; end < text.size(); ++end) {
                auto child = trie_[node].next.find(static_cast<unsigned char>(text[end]));
                if (child == trie_[node].next.end()) break;
                node = child->second;
                if (trie_[node].id >= 0) {
                    int id = trie_[node].id;
                    update(end + 1, id, scores_[id]);
                    if (end + 1 - start == width) single = true;
                }
            }
            if (!single) update(start + width, unk_, min_score_ - 10.0);
            start += width;
        }
        struct Piece { size_t start, end; int id; };
        std::vector<Piece> path;
        for (size_t end = text.size(); end;) {
            auto b = best[end];
            if (b.start == std::string::npos) throw std::runtime_error("Invalid Unigram path");
            path.push_back({b.start, end, b.id}); end = b.start;
        }
        std::reverse(path.begin(), path.end());
        for (size_t i = 0; i < path.size(); ++i) {
            auto piece = path[i];
            if (piece.id == unk_) while (i + 1 < path.size() && path[i + 1].id == unk_) piece.end = path[++i].end;
            auto value = text.substr(piece.start, piece.end - piece.start);
            auto known = ids_.find(value);
            if (known != ids_.end()) result.push_back(known->second);
            else {
                bool fallback = std::all_of(value.begin(), value.end(), [&](unsigned char b) { return bytes_[b] >= 0; });
                if (fallback) for (unsigned char b : value) result.push_back(bytes_[b]);
                else result.push_back(unk_);
            }
        }
    }
    void ordinary(const std::string& raw, std::vector<int>& ids) const {
        if (raw.empty()) return;
        std::string text = marker; // Prepend normalizer.
        for (char c : raw) { if (c == ' ') text += marker; else text += c; }
        // Metaspace: split with the delimiter merged into the following piece.
        size_t start = 0;
        while (start < text.size()) {
            auto end = text.find(marker, start + marker.size());
            if (end == std::string::npos) end = text.size();
            unigram(text.substr(start, end - start), ids); start = end;
        }
    }
public:
    explicit Tokenizer(const std::string& path) {
        std::ifstream file(path);
        if (!file) throw std::runtime_error("Cannot open tokenizer: " + path);
        Json j; file >> j;
        const auto& model = j.at("model");
        if (model.at("type") != "Unigram" || !model.at("byte_fallback").get<bool>())
            throw std::runtime_error("Expected a byte-fallback Unigram tokenizer");
        check_normalizer(j.at("normalizer"));
        const auto& pre = j.at("pre_tokenizer");
        if (pre.at("type") != "Metaspace" || pre.at("replacement") != marker ||
            pre.at("prepend_scheme") != "always" || !pre.at("split").get<bool>() || !j.at("post_processor").is_null())
            throw std::runtime_error("Unsupported JSON tokenizer pipeline");
        const auto& dec = j.at("decoder");
        if (dec.at("type") != "Sequence" || dec.at("decoders").size() != 3 ||
            dec.at("decoders")[0] != pre || dec.at("decoders")[1].at("type") != "ByteFallback" ||
            dec.at("decoders")[2].at("type") != "Fuse") throw std::runtime_error("Unsupported JSON decoder");
        for (const auto& item : model.at("vocab")) {
            int id = static_cast<int>(vocab_.size());
            vocab_.push_back(item[0]); scores_.push_back(item[1]); ids_[vocab_.back()] = id;
            min_score_ = std::min(min_score_, scores_.back());
            size_t node = 0;
            for (unsigned char byte : vocab_.back()) {
                auto child = trie_[node].next.find(byte);
                if (child == trie_[node].next.end()) {
                    size_t next = trie_.size(); trie_[node].next[byte] = next;
                    trie_.push_back({}); node = next;
                } else node = child->second;
            }
            trie_[node].id = id;
        }
        unk_ = model.at("unk_id"); special_.resize(vocab_.size());
        for (const auto& token : j.at("added_tokens")) {
            if (token.at("single_word") != false || token.at("lstrip") != false ||
                token.at("rstrip") != false || token.at("normalized") != false)
                throw std::runtime_error("Unsupported added-token matching options");
            int id = token.at("id");
            if (id < 0 || size_t(id) >= vocab_.size() || vocab_[id] != token.at("content"))
                throw std::runtime_error("Added token is outside the vocabulary");
            added_.push_back({vocab_[id], id}); special_[id] = token.at("special");
        }
        for (unsigned byte = 0; byte < 256; ++byte) {
            char name[7]; std::snprintf(name, sizeof name, "<0x%02X>", byte);
            auto found = ids_.find(name); bytes_[byte] = found == ids_.end() ? -1 : found->second;
        }
    }
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> result;
        size_t start = 0;
        while (start < text.size()) {
            size_t at = std::string::npos; const std::pair<std::string, int>* match = nullptr;
            for (const auto& token : added_) {
                size_t found = text.find(token.first, start);
                if (found < at || (found == at && match && token.first.size() > match->first.size())) {
                    at = found; match = &token;
                }
            }
            if (!match) { ordinary(text.substr(start), result); break; }
            ordinary(text.substr(start, at - start), result);
            result.push_back(match->second); start = at + match->first.size();
        }
        return result;
    }
    std::string decode(const std::vector<int>& ids) const {
        std::string output, pending;
        auto flush = [&] {
            if (pending.empty()) return;
            bool valid = true;
            try { characters(pending); } catch (const std::exception&) { valid = false; }
            if (valid) output += pending;
            else for (size_t i = 0; i < pending.size(); ++i) output += replacement;
            pending.clear();
        };
        size_t index = 0;
        for (int id : ids) {
            if (id < 0 || size_t(id) >= vocab_.size()) continue;
            if (special_[id]) continue;
            std::string token;
            for (const auto& c : characters(vocab_[id])) {
                if (c.code == 0x2581) { if (index != 0) token += ' '; }
                else token += vocab_[id].substr(c.start, c.size);
            }
            ++index;
            auto hex = [](char c) { return c >= '0' && c <= '9' ? c - '0' :
                c >= 'A' && c <= 'F' ? c - 'A' + 10 : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1; };
            bool byte = token.size() == 6 && token.compare(0, 3, "<0x") == 0 && token.back() == '>' &&
                        hex(token[3]) >= 0 && hex(token[4]) >= 0;
            if (byte) {
                pending += static_cast<char>(hex(token[3]) * 16 + hex(token[4]));
            } else { flush(); output += token; }
        }
        flush(); return output;
    }
};

std::vector<std::pair<size_t, std::string>> segments(const Tokenizer& tok,
    const std::vector<int>& ids, const std::vector<int>& marks, bool decimal) {
    std::vector<size_t> boundaries{0};
    bool previous = false;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (std::find(marks.begin(), marks.end(), ids[i]) != marks.end()) previous = true;
        else {
            if (previous) {
                bool number = false;
                if (decimal) {
                    auto left = characters(tok.decode(std::vector<int>(ids.begin(), ids.begin() + i)));
                    auto right = characters(tok.decode(std::vector<int>(ids.begin() + i, ids.end())));
                    number = left.size() >= 2 && left.back().code == '.' && digit(left[left.size()-2].code) &&
                             !right.empty() && digit(right.front().code);
                }
                if (!number) boundaries.push_back(i);
            }
            previous = false;
        }
    }
    boundaries.push_back(ids.size());
    std::vector<std::pair<size_t, std::string>> result;
    for (size_t i = 1; i < boundaries.size(); ++i)
        result.push_back({boundaries[i] - boundaries[i-1], tok.decode(std::vector<int>(ids.begin() + boundaries[i-1], ids.begin() + boundaries[i]))});
    return result;
}

Json process(const Tokenizer& tok, const std::string& text, const Json& config) {
    if (config.value("encode_only", false)) return {{"ids", tok.encode(text)}};
    auto prepared = prepare(text, config).first;
    auto ends = tok.encode(".!...?"); ends.erase(ends.begin());
    auto commas = tok.encode(",;:"); commas.erase(commas.begin());
    size_t limit = config.value("max_tokens", size_t(50));
    auto parts = segments(tok, tok.encode(trim(prepared)), ends, true);
    std::vector<std::pair<size_t, std::string>> refined;
    for (const auto& part : parts) {
        if (part.first > limit) {
            auto sub = segments(tok, tok.encode(trim(part.second)), commas, false);
            if (sub.size() > 1) { refined.insert(refined.end(), sub.begin(), sub.end()); continue; }
        }
        refined.push_back(part);
    }
    std::vector<std::string> chunks;
    std::string current;
    size_t count = 0;
    for (const auto& part : refined) {
        if (!current.empty() && count + part.first > limit) { chunks.push_back(trim(current)); current.clear(); count = 0; }
        if (!current.empty()) current += ' ';
        current += part.second; count += part.first;
    }
    if (!current.empty()) chunks.push_back(trim(current));
    Json result = Json::array();
    for (const auto& chunk : chunks) {
        auto p = prepare(chunk, config);
        int extra = p.second + 2;
        auto recommended = config.find("model_recommended_frames_after_eos");
        if (recommended != config.end() && recommended->is_number_unsigned()) extra = recommended->get<int>();
        result.push_back({{"text", p.first}, {"ids", tok.encode(p.first)}, {"eos_extra", extra}});
    }
    return {{"chunks", result}};
}
} // namespace pocket_text

extern "C" {
void* pocket_tokenizer_new(const char* path) {
    try { return new pocket_text::Tokenizer(path); } catch (const std::exception&) { return nullptr; }
}
void pocket_tokenizer_free(void* tokenizer) { delete static_cast<pocket_text::Tokenizer*>(tokenizer); }
void pocket_text_free(char* text) { delete[] text; }
char* pocket_text_process(const void* tokenizer, const char* text, const char* config_json) {
    std::string result;
    try {
        if (!tokenizer || !text || !config_json) throw std::runtime_error("Null tokenizer input");
        result = pocket_text::process(*static_cast<const pocket_text::Tokenizer*>(tokenizer), text,
                                     pocket_text::Json::parse(config_json)).dump();
    } catch (const std::exception& error) { result = pocket_text::Json({{"error", error.what()}}).dump(); }
    auto output = new char[result.size() + 1]; std::memcpy(output, result.c_str(), result.size() + 1); return output;
}
}
