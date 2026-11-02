#include "normalizer.h"
#include "numbertext/Soros.hxx"
#include "pocket_number_rules.h"
#include "../native_tokenizer/unicode_data.hpp"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <vector>

namespace pocket_speech {
namespace {
bool digit(char c) { return c >= '0' && c <= '9'; }
bool alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool word_at(const std::string& s, size_t pos) {
    unsigned char c = s[pos];
    if (c < 0x80) return alpha(c) || digit(c) || c == '_';
    // Latin accents remain part of words, while common Unicode spaces and
    // punctuation must be valid boundaries (including French guillemets).
    unsigned n = c < 0xE0 ? 1 : c < 0xF0 ? 2 : 3;
    uint32_t cp = c & (0x7F >> (n+1));
    for (unsigned j=1; j<=n; ++j) cp=(cp<<6)|(static_cast<unsigned char>(s[pos+j])&0x3F);
    return !(cp == 0x85 || cp == 0xA0 || cp == 0xAB || cp == 0xBB || cp == 0xB0 || cp == 0xB7 ||
        (cp >= 0x2000 && cp <= 0x200A) || (cp >= 0x2010 && cp <= 0x202F) || cp == 0x205F || cp == 0x3000);
}
bool word_before(const std::string& s, size_t pos) {
    if (!pos) return false;
    --pos;
    while (pos && (static_cast<unsigned char>(s[pos])&0xC0) == 0x80) --pos;
    return word_at(s,pos);
}
bool boundary(const std::string& s, size_t end) { return end == s.size() || !word_at(s,end); }
std::wstring wide(const std::string& s) {
    std::wstring out;
    for (size_t i = 0; i < s.size();) {
        unsigned char first = s[i++];
        unsigned n = first < 0x80 ? 0 : first >= 0xC2 && first <= 0xDF ? 1 :
            first >= 0xE0 && first <= 0xEF ? 2 : first >= 0xF0 && first <= 0xF4 ? 3 : 4;
        if (n == 4 || i+n > s.size()) throw std::runtime_error("Invalid UTF-8 text");
        uint32_t cp = n ? first & (0x7F >> (n+1)) : first;
        for (unsigned j = 0; j < n; ++j) {
            unsigned char c = s[i++];
            if ((c & 0xC0) != 0x80) throw std::runtime_error("Invalid UTF-8 text");
            cp = (cp << 6) | (c & 0x3F);
        }
        if ((n == 1 && cp < 0x80) || (n == 2 && cp < 0x800) || (n == 3 && cp < 0x10000) ||
            cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) throw std::runtime_error("Invalid UTF-8 text");
        if (sizeof(wchar_t) == 2 && cp > 0xFFFF) {
            cp -= 0x10000; out += wchar_t(0xD800+(cp>>10)); out += wchar_t(0xDC00+(cp&0x3FF));
        } else out += wchar_t(cp);
    }
    return out;
}
std::string utf8(const std::wstring& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        uint32_t cp = s[i];
        if (sizeof(wchar_t) == 2 && cp >= 0xD800 && cp <= 0xDBFF && i+1 < s.size()) {
            uint32_t low = s[++i];
            if (low < 0xDC00 || low > 0xDFFF) throw std::runtime_error("Invalid UTF-16 in number rules");
            cp = 0x10000 + ((cp-0xD800)<<10) + low-0xDC00;
        }
        if (cp < 0x80) out += char(cp);
        else if (cp < 0x800) { out += char(0xC0|(cp>>6)); out += char(0x80|(cp&0x3F)); }
        else if (cp < 0x10000) { out += char(0xE0|(cp>>12)); out += char(0x80|((cp>>6)&0x3F)); out += char(0x80|(cp&0x3F)); }
        else { out += char(0xF0|(cp>>18)); out += char(0x80|((cp>>12)&0x3F)); out += char(0x80|((cp>>6)&0x3F)); out += char(0x80|(cp&0x3F)); }
    }
    return out;
}
void append(std::string& s, const std::string& part) { if (!s.empty()) s += ' '; s += part; }
void replace_all(std::string& s, const std::string& a, const std::string& b) {
    for (size_t p = 0; (p = s.find(a, p)) != std::string::npos; p += b.size()) s.replace(p, a.size(), b);
}

struct Entry {
    std::string source, replacement;
    std::optional<std::regex> pattern;
    std::optional<std::regex> non_initial;
    std::string folded_literal;
};
using Entries = std::vector<Entry>;
std::string uppercase_rune(const std::string& s, size_t pos, size_t& width) {
    unsigned char first = s[pos];
    width = first < 0x80 ? 1 : first < 0xE0 ? 2 : first < 0xF0 ? 3 : 4;
    if (first < 0x80) return std::string(1,first >= 'a' && first <= 'z' ? first-32 : first);
    uint32_t cp = first & (0x7F >> width);
    for (size_t j=1; j<width; ++j) cp=(cp<<6)|(static_cast<unsigned char>(s[pos+j])&0x3F);
    const auto& table = pocket_text::uppercase;
    auto it = std::lower_bound(std::begin(table),std::end(table),cp,
        [](const pocket_text::Uppercase& item,uint32_t value) { return item.code < value; });
    return it != std::end(table) && it->code == cp ? it->text : s.substr(pos,width);
}
std::string uppercase_text(const std::string& s) {
    std::string out;
    for (size_t i=0; i<s.size();) { size_t width; out += uppercase_rune(s,i,width); i+=width; }
    return out;
}
size_t literal_length(const Entry& entry,const std::string& s,size_t pos) {
    if (entry.folded_literal.empty()) return s.compare(pos,entry.source.size(),entry.source) == 0 ? entry.source.size() : 0;
    size_t at=pos, matched=0;
    while (at<s.size() && matched<entry.folded_literal.size()) {
        size_t width; auto part=uppercase_rune(s,at,width);
        if (entry.folded_literal.compare(matched,part.size(),part) != 0) return 0;
        at+=width; matched+=part.size();
    }
    return matched == entry.folded_literal.size() ? at-pos : 0;
}
std::string disable_start_anchors(const std::string& pattern) {
    // With match_prev_avail, libc++ can still treat a searched suffix as BOL.
    // Outside the original start, replace anchor assertions with an always-
    // false, noncapturing assertion; keep literal/character-class carets intact.
    std::string result;
    bool escaped = false, character_class = false;
    for (size_t i = 0; i < pattern.size(); ++i) {
        char c = pattern[i];
        if (escaped) { result += c; escaped = false; }
        else if (c == '\\') { result += c; escaped = true; }
        else if (character_class && c == '[' && i+1 < pattern.size() &&
                 std::string(":.=").find(pattern[i+1]) != std::string::npos) {
            // POSIX classes/collating elements can contain their own closing
            // bracket, e.g. [[:alpha:]^]. It does not end the outer class.
            auto end = pattern.find(std::string(1,pattern[i+1])+"]",i+2);
            if (end == std::string::npos) result += c;
            else { result += pattern.substr(i,end+2-i); i=end+1; }
        }
        else if (c == '[' && !character_class) { result += c; character_class = true; }
        else if (c == ']' && character_class) { result += c; character_class = false; }
        else if (c == '^' && !character_class) result += "(?!)";
        else result += c;
    }
    return result;
}
Entries read_csv(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot read lexicon: " + path);
    std::string data((std::istreambuf_iterator<char>(f)), {});
    if (data.compare(0, 3, "\xEF\xBB\xBF") == 0) data.erase(0, 3);
    // Validate UTF-8 before any partial substitutions can be used.
    wide(data);
    Entries entries;
    std::set<std::string> seen;
    size_t i = 0, row = 0;
    while (i < data.size()) {
        ++row;
        if (data[i] == '#') { while (i < data.size() && data[i] != '\n') ++i; if (i < data.size()) ++i; continue; }
        if (data[i] == '\n' || data[i] == '\r') { if (data[i++] == '\r' && i < data.size() && data[i] == '\n') ++i; continue; }
        std::vector<std::string> fields;
        bool more = true;
        while (more) {
            std::string field;
            if (i < data.size() && data[i] == '"') {
                ++i; bool closed = false;
                while (i < data.size()) {
                    char c = data[i++];
                    if (c != '"') field += c;
                    else if (i < data.size() && data[i] == '"') { field += '"'; ++i; }
                    else { closed = true; break; }
                }
                if (!closed) throw std::runtime_error("Unclosed quote in lexicon: " + path);
            } else {
                while (i < data.size() && data[i] != ',' && data[i] != '\r' && data[i] != '\n') {
                    if (data[i] == '"') throw std::runtime_error("Unexpected quote in lexicon: " + path);
                    field += data[i++];
                }
            }
            fields.push_back(field);
            more = i < data.size() && data[i] == ',';
            if (more) ++i;
            else if (i < data.size()) {
                if (data[i] != '\r' && data[i] != '\n') throw std::runtime_error("Invalid CSV delimiter in lexicon: " + path);
                if (data[i++] == '\r' && i < data.size() && data[i] == '\n') ++i;
            }
        }
        if (fields.size() != 2 || fields[0].empty())
            throw std::runtime_error("Expected source,replacement in lexicon " + path + " at record " + std::to_string(row));
        if (entries.empty() && fields[0] == "source" && fields[1] == "replacement") continue;
        if (!seen.insert(fields[0]).second) throw std::runtime_error("Duplicate lexicon source: " + fields[0]);
        Entry entry{fields[0],fields[1],std::nullopt,std::nullopt,""};
        bool regex = entry.source.compare(0,2,"/r") == 0 || entry.source.compare(0,2,"/R") == 0;
        bool insensitive = entry.source.compare(0,2,"/i") == 0 || entry.source.compare(0,2,"/R") == 0;
        if (entry.source.compare(0,2,"//") == 0) entry.source.erase(0,1); // escaped literal leading slash
        else if (regex || insensitive) entry.source.erase(0,2);
        else if (entry.source[0] == '/') throw std::runtime_error("Unknown lexicon prefix in " + path +
            " at record " + std::to_string(row) + "; use // for a literal leading slash");
        if (entry.source.empty()) throw std::runtime_error(std::string("Empty lexicon ") + (regex ? "regex" : "text") +
            " in " + path + " at record " + std::to_string(row));
        if (regex) {
            try {
                auto pattern = entry.source;
                auto flags = std::regex::ECMAScript;
                if (insensitive) flags |= std::regex::icase;
                entry.pattern.emplace(pattern,flags);
                auto non_initial = disable_start_anchors(pattern);
                if (non_initial != pattern) entry.non_initial.emplace(non_initial,flags);
            }
            catch (const std::regex_error& e) {
                throw std::runtime_error("Invalid lexicon regex in " + path + " at record " + std::to_string(row) +
                    " (" + fields[0] + "): " + e.what());
            }
        } else if (insensitive) entry.folded_literal = uppercase_text(entry.source);
        entries.push_back(std::move(entry));
    }
    return entries;
}

struct Unit { const char* suffix; const char* en; const char* ens; const char* fr; const char* frs; bool feminine = false; };
const Unit units[] = {
    {"Mbps", "megabit per second", "megabits per second", "mégabit par seconde", "mégabits par seconde"},
    {"mbps", "megabit per second", "megabits per second", "mégabit par seconde", "mégabits par seconde"},
    {"Gbps", "gigabit per second", "gigabits per second", "gigabit par seconde", "gigabits par seconde"},
    {"gbps", "gigabit per second", "gigabits per second", "gigabit par seconde", "gigabits par seconde"},
    {"kbps", "kilobit per second", "kilobits per second", "kilobit par seconde", "kilobits par seconde"},
    {"Kbps", "kilobit per second", "kilobits per second", "kilobit par seconde", "kilobits par seconde"},
    {"MB/s", "megabyte per second", "megabytes per second", "mégaoctet par seconde", "mégaoctets par seconde"},
    {"km/h", "kilometer per hour", "kilometers per hour", "kilomètre par heure", "kilomètres par heure"},
    {"mph", "mile per hour", "miles per hour", "mile par heure", "miles par heure"},
    {"kHz", "kilohertz", "kilohertz", "kilohertz", "kilohertz"},
    {"MHz", "megahertz", "megahertz", "mégahertz", "mégahertz"},
    {"GHz", "gigahertz", "gigahertz", "gigahertz", "gigahertz"},
    {"mins", "minute", "minutes", "minute", "minutes", true},
    {"min", "minute", "minutes", "minute", "minutes", true},
    {"hrs", "hour", "hours", "heure", "heures", true},
    {"hr", "hour", "hours", "heure", "heures", true},
    {"lbs", "pound", "pounds", "livre", "livres", true},
    {"lb", "pound", "pounds", "livre", "livres", true},
    {"oz", "ounce", "ounces", "once", "onces", true},
    {"kg", "kilogram", "kilograms", "kilogramme", "kilogrammes"},
    {"mg", "milligram", "milligrams", "milligramme", "milligrammes"},
    {"km", "kilometer", "kilometers", "kilomètre", "kilomètres"},
    {"cm", "centimeter", "centimeters", "centimètre", "centimètres"},
    {"mm", "millimeter", "millimeters", "millimètre", "millimètres"},
    {"ms", "millisecond", "milliseconds", "milliseconde", "millisecondes", true},
    {"MB", "megabyte", "megabytes", "mégaoctet", "mégaoctets"},
    {"GB", "gigabyte", "gigabytes", "gigaoctet", "gigaoctets"},
    {"KB", "kilobyte", "kilobytes", "kilooctet", "kilooctets"},
    {"Hz", "hertz", "hertz", "hertz", "hertz"},
    {"mL", "milliliter", "milliliters", "millilitre", "millilitres"},
    {"ml", "milliliter", "milliliters", "millilitre", "millilitres"},
    {"h", "hour", "hours", "heure", "heures", true},
    {"s", "second", "seconds", "seconde", "secondes", true},
    {"g", "gram", "grams", "gramme", "grammes"},
    {"m", "meter", "meters", "mètre", "mètres"},
    {"L", "liter", "liters", "litre", "litres"},
    {"l", "liter", "liters", "litre", "litres"},
    {"K", "kelvin", "kelvins", "kelvin", "kelvins"},
    {"%", "percent", "percent", "pour cent", "pour cent"},
};
}

struct Normalizer::Impl {
    bool enabled, french;
    Entries lexicon;
    std::regex numeric;

    Impl(const std::string& language, const std::string& path, bool on)
        : enabled(on), french(language == "french" || language == "fr"),
          numeric(french ? R"([+-]?(?:[0-9]{1,3}(?: [0-9]{3})+(?![0-9])|[0-9]+)(?:[,.][0-9]+)?)"
                         : R"([+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+(?![0-9])|[0-9]+)(?:\.[0-9]+)?)") {
        if (!enabled) return;
        if (language != "english" && language != "en" && !french)
            throw std::runtime_error("Speech normalization supports english/french; disable it for language: " + language);
        std::string file = path;
        if (file.empty()) {
            file = std::string("lexicon.") + (french ? "french" : "english") + ".csv";
            if (!std::filesystem::exists(file)) file.clear();
        }
        if (!file.empty() && file != "-") lexicon = read_csv(file);
    }

    std::string spell(const std::string& value, const std::string& mode = "") const {
        // Upstream run() only reads compiled rules and keeps recursion state in
        // its arguments. Share them across engines; do not recompile per request.
        auto rules = [](bool french) -> Soros& {
            if (french) { static Soros fr(wide(french_rules), L"fr"); return fr; }
            static Soros en(wide(english_rules), L"en"); return en;
        };
        auto input = wide(mode.empty() ? value : mode + " " + value);
        if (rules(french).run(input) < 0 || input.empty())
            throw std::runtime_error("Number spelling failed: " + value);
        std::string out = utf8(input);
        // Numbertext's English scale separators are written commas; they should
        // not become sentence/chunk boundaries in the speech engine.
        replace_all(out, ",", "");
        return out;
    }
    std::string digits(const std::string& value, bool letters = false) const {
        static const char* en[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
        static const char* fr[] = {"zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"};
        static const char* en_letters[] = {"ay", "bee", "see", "dee", "ee", "eff", "jee", "aitch", "eye", "jay", "kay", "el", "em", "en", "oh", "pee", "cue", "ar", "ess", "tee", "you", "vee", "double you", "ex", "why", "zee"};
        static const char* fr_letters[] = {"a", "bé", "cé", "dé", "e", "effe", "gé", "ache", "i", "ji", "ka", "elle", "emme", "enne", "o", "pé", "ku", "erre", "esse", "té", "u", "vé", "double vé", "ixe", "i grec", "zède"};
        std::string out;
        for (char c : value) {
            if (digit(c)) append(out, (french ? fr : en)[c-'0']);
            else if (letters && alpha(c)) append(out, (french ? fr_letters : en_letters)[(c >= 'a' ? c-'a' : c-'A')]);
            else if (letters && c == '-' && !out.empty() && out.back() != ',') out += ',';
            else if (letters && c == '_') append(out, french ? "tiret bas" : "underscore");
        }
        return out;
    }
    std::string phone_digits(const std::string& value) const {
        std::string out;
        for (size_t i=0; i<value.size();) {
            if (!digit(value[i])) { ++i; continue; }
            size_t end=i+1;
            while (end<value.size() && digit(value[end])) ++end;
            if (!out.empty()) out += ", ";
            out += digits(value.substr(i,end-i));
            i=end;
        }
        return out;
    }
    // Canonical decimal uses '.', with no grouping separators. Never use float:
    // large integers, trailing decimal zeros and identifiers must remain exact.
    std::string canonical(std::string s) const {
        if (french) { replace_all(s, " ", ""); replace_all(s, ",", "."); }
        else replace_all(s, ",", "");
        return s;
    }
    std::string number(const std::string& value, bool feminine = false) const {
        std::string s = value, prefix;
        if (s[0] == '-' || s[0] == '+') { prefix = s[0] == '-' ? (french ? "moins " : "minus ") : "plus "; s.erase(0,1); }
        size_t point = s.find('.');
        std::string whole = s.substr(0, point);
        bool sequence = (whole.size() > 1 && whole[0] == '0') || whole.size() > 18;
        std::string out = sequence ? digits(whole) : spell(whole, french && feminine ? "feminine" : "");
        // Upstream's feminine rule only handles standalone 1. Agree compounds
        // such as vingt-et-une heures as well, without altering mille/un million.
        if (french && feminine && out.size() >= 3 && out.compare(out.size()-3,3,"-un") == 0) out += 'e';
        if (point != std::string::npos) out += (french ? " virgule " : " point ") + digits(s.substr(point+1));
        return prefix + out;
    }
    std::string year_or_number(const std::string& value) const {
        if (!french && value.size() == 4 &&
            std::all_of(value.begin(),value.end(),digit) &&
            (value[0] == '1' || value.compare(0,2,"20") == 0)) {
            auto first = value.substr(0,2), last = value.substr(2);
            if (value == "1000") return "one thousand";
            if (first == "20" && last[0] == '0')
                return last == "00" ? "two thousand" : "two thousand " + spell(last);
            if (last == "00") return spell(first) + " hundred";
            if (last[0] == '0') return spell(first) + " oh " + spell(last.substr(1));
            return spell(first) + " " + spell(last);
        }
        return number(value);
    }
    std::string read_token(const std::string& value, bool compound_tail = false) const {
        bool has_digit = std::any_of(value.begin(), value.end(), digit);
        bool has_letter = std::any_of(value.begin(), value.end(), alpha);
        bool lower = std::any_of(value.begin(), value.end(), [](char c) { return c >= 'a' && c <= 'z'; });
        bool punctuation = value.find_first_of("-_()") != std::string::npos;
        if (!has_digit && !compound_tail) return value;
        // Previous compact uppercase-ID policy, kept for easy restoration:
        // if (!has_letter || (!lower && !punctuation)) return digits(value, true);
        if (!has_letter) return digits(value,true);
        if (!lower && !punctuation) {
            if (!has_digit) return digits(value,true); // a letter in 3-D-printer
            std::string out;
            for (size_t i = 0; i < value.size();) {
                bool numeric = digit(value[i]);
                size_t end = i+1;
                while (end < value.size() && digit(value[end]) == numeric) ++end;
                auto part = value.substr(i,end-i);
                if (numeric) {
                    // Read numeric runs as values, not years or individual
                    // digits; keep alphabetic runs intact (IN2BACH -> IN 2 BACH).
                    auto first = part.find_first_not_of('0');
                    part = number(first == std::string::npos ? "0" : part.substr(first));
                }
                append(out,part); i=end;
            }
            return out;
        }
        if (!punctuation) return value; // lowercase/mixed-case words are not IDs

        std::vector<std::pair<std::string,bool>> parts;
        bool parenthesized = false, mixed_part = false;
        for (size_t i = 0; i < value.size();) {
            if (value[i] == '(') { parenthesized = true; ++i; continue; }
            if (value[i] == ')') { parenthesized = false; ++i; continue; }
            if (value[i] == '-' || value[i] == '_') { ++i; continue; }
            auto end = value.find_first_of("-_()", i);
            if (end == std::string::npos) end = value.size();
            auto part = value.substr(i,end-i);
            mixed_part |= std::any_of(part.begin(),part.end(),digit) && std::any_of(part.begin(),part.end(),alpha);
            parts.emplace_back(part,parenthesized); i = end;
        }
        // Preserve structured all-caps codes such as R2-D2 and 87M-34KEL6.
        // A word compound such as 3-D-printer must not take this path.
        if (!lower && mixed_part) return digits(value,true);
        bool numeric_groups = !lower && std::count_if(parts.begin(),parts.end(),[](const auto& p) {
            return std::all_of(p.first.begin(),p.first.end(),digit);
        }) > 1;
        std::string out;
        for (const auto& item : parts) {
            const auto& part = item.first;
            bool numeric_part = std::all_of(part.begin(),part.end(),digit);
            bool upper_id = std::any_of(part.begin(),part.end(),digit) &&
                std::all_of(part.begin(),part.end(),[](char c) { return digit(c) || (c >= 'A' && c <= 'Z'); });
            if (numeric_groups && !out.empty()) out += ',';
            if (numeric_part) append(out,item.second || numeric_groups ? digits(part) : number(part));
            else if (upper_id || (part.size() == 1 && part[0] >= 'A' && part[0] <= 'Z')) append(out,digits(part,true));
            else append(out,part);
        }
        return out;
    }
    bool singular(std::string s) const {
        if (s[0] == '+' || s[0] == '-') s.erase(0,1);
        auto dot = s.find('.');
        if (dot != std::string::npos) { if (s.find_first_not_of('0', dot+1) != std::string::npos) return false; s.resize(dot); }
        auto first = s.find_first_not_of('0');
        return first != std::string::npos && s.substr(first) == "1";
    }
    std::string substitutions(const std::string& s) const {
        std::string out;
        for (size_t i = 0; i <= s.size();) {
            bool found = false;
            size_t length = 0;
            std::string replacement;
            for (const auto& entry : lexicon) {
                if (entry.pattern) {
                    std::smatch match;
                    auto flags = std::regex_constants::match_continuous;
                    // Keep word boundaries relative to the original input.
                    // The compiled non_initial variant also handles ^ correctly
                    // on standard libraries that reset BOL with match_prev_avail.
                    if (i) flags |= std::regex_constants::match_prev_avail;
                    const auto& pattern = i && entry.non_initial ? *entry.non_initial : *entry.pattern;
                    // Previous longest-match guard (restore and remove break):
                    // && (!found || size_t(match.length()) > length)
                    if (std::regex_search(s.cbegin()+i,s.cend(),match,pattern,flags)) {
                        length = match.length(); replacement = match.format(entry.replacement); found = true;
                        break; // first matching CSV row wins
                    }
                } else if (!word_before(s,i)) {
                    size_t matched = literal_length(entry,s,i);
                    // Previous longest-match guard (restore and remove break):
                    // (!found || matched > length) &&
                    if (matched && boundary(s,i+matched)) {
                        length = matched; replacement = entry.replacement; found = true;
                        break; // first matching CSV row wins
                    }
                }
            }
            if (found) { out += replacement; if (length) { i += length; continue; } }
            if (i == s.size()) break;
            // Empty regex matches insert once at this position. Copy an entire
            // UTF-8 character to advance without looping or splitting a rune.
            unsigned char c = s[i];
            size_t width = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
            out.append(s,i,width); i+=width;
        }
        // std::regex works on UTF-8 bytes; a capture must not leave half a rune.
        try { wide(out); } catch (const std::exception&) {
            throw std::runtime_error("Lexicon substitutions produced invalid UTF-8 text");
        }
        return out;
    }
    std::string scale_at(const std::string& s, size_t& end) const {
        size_t at = end;
        while (at < s.size() && (s[at] == ' ' || s[at] == '\t')) ++at;
        static const std::regex magnitude(R"((thousand|million|milliard|billion|trillion|mille)s?)", std::regex::icase);
        std::smatch m;
        if (std::regex_search(s.cbegin()+at,s.cend(),m,magnitude,std::regex_constants::match_continuous) && boundary(s,at+m.length())) {
            auto result = m[1].str();
            std::transform(result.begin(),result.end(),result.begin(),[](char c){ return c >= 'A' && c <= 'Z' ? c+32 : c; });
            end = at + m.length(); return result;
        }
        if (at < s.size() && std::string("KkMmBbTt").find(s[at]) != std::string::npos && boundary(s,at+1)) {
            end = at+1; return s.substr(at,1);
        }
        return "";
    }
    std::string money(const std::string& value, const std::string& symbol, const std::string& scale) const {
        bool one = singular(value) && scale.empty();
        std::string unit = symbol == "$" ? (one ? "dollar" : "dollars") : symbol == "€" ? (one ? "euro" : "euros") :
            french ? (one ? "livre sterling" : "livres sterling") : (one ? "pound" : "pounds");
        auto point = value.find('.');
        if (scale.empty() && point != std::string::npos && value.size() - point == 3) {
            std::string whole = value.substr(0, point), cents = value.substr(point+1);
            unit = symbol == "$" ? (singular(whole) ? "dollar" : "dollars") : symbol == "€" ? (singular(whole) ? "euro" : "euros") :
                french ? (singular(whole) ? "livre sterling" : "livres sterling") : (singular(whole) ? "pound" : "pounds");
            std::string result = number(whole, french && symbol == "£") + " " + unit;
            if (cents != "00") {
                auto pos = cents.find_first_not_of('0'); cents = pos == std::string::npos ? "0" : cents.substr(pos);
                result += (french ? " et " : " and ") + number(cents) + " ";
                result += french ? (singular(cents) ? "centime" : "centimes") : symbol == "£" ? (singular(cents) ? "penny" : "pence") : (singular(cents) ? "cent" : "cents");
            }
            return result;
        }
        std::string magnitude;
        if (!scale.empty()) {
            if (scale == "K" || scale == "k" || scale == "thousand" || scale == "mille") magnitude = french ? " mille" : " thousand";
            else if (scale == "M" || scale == "m" || scale == "million") magnitude = french ? (singular(value) ? " million" : " millions") : " million";
            else if (scale == "trillion") magnitude = french ? (singular(value) ? " trillion" : " trillions") : " trillion";
            else if (scale == "T" || scale == "t") magnitude = french ? (singular(value) ? " billion" : " billions") : " trillion";
            else if (french && scale == "billion") magnitude = singular(value) ? " billion" : " billions";
            else magnitude = french ? (singular(value) ? " milliard" : " milliards") : " billion";
        }
        auto spoken = number(value, french && symbol == "£" && scale.empty());
        if (french && magnitude == " mille" && value == "1") { spoken = "mille"; magnitude.clear(); }
        auto join = french && !scale.empty() && !magnitude.empty() && magnitude != " mille" ?
            (symbol == "€" ? " d'" : " de ") : " ";
        return spoken + magnitude + join + unit;
    }

    static std::string unpad(const std::string& value) {
        auto at = value.find_first_not_of('0');
        return at == std::string::npos ? "0" : value.substr(at);
    }
    static bool valid_date(int month, int day, int year) {
        static const int days[] = {31,28,31,30,31,30,31,31,30,31,30,31};
        if (month < 1 || month > 12 || day < 1 || year < 1) return false;
        bool leap = year%4 == 0 && (year%100 != 0 || year%400 == 0);
        return day <= days[month-1] + (month == 2 && leap);
    }
    // Return a complete contextual reading; never feed generated words back
    // through normalization. This also keeps abbreviation dots out of chunking.
    bool date_time(const std::string& s, size_t at, size_t& end, std::string& spoken) const {
        if (!digit(s[at]) && (french || !alpha(s[at]))) return false;
        auto match = [&](const std::regex& re, std::smatch& m) {
            if (!std::regex_search(s.cbegin()+at,s.cend(),m,re,std::regex_constants::match_continuous)) return false;
            size_t after = at+m.length();
            if (!boundary(s,after)) return false;
            return after == s.size() || (s[after] != '-' && s[after] != '/' && s[after] != '_' &&
                !(s[after] == '.' && after+1 < s.size() && digit(s[after+1])));
        };
        static const std::string months = "January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|August|Aug|September|Sept|Sep|October|Oct|November|Nov|December|Dec";
        static const std::string weekday = R"((?:(Monday|Mon|Tuesday|Tues|Tue|Wednesday|Weds|Wed|Thursday|Thurs|Thur|Thu|Friday|Fri|Saturday|Sat|Sunday|Sun)\.?[ \t]*,?[ \t]+)?)";
        static const std::string year = R"((?:[ \t]*,[ \t]*|[ \t]+)([0-9]{4}|[0-9]{2})(?![0-9]))";
        static const std::regex month_first(weekday+"("+months+R"()\.?[ \t]+([0-9]{1,2})(?:st|nd|rd|th)?(?:)"+year+")?",std::regex::icase);
        static const std::regex named_day_first(weekday+R"(([0-9]{1,2})(?:st|nd|rd|th)?(?:[ \t]*-[ \t]*|[ \t]+)()"+months+R"()\.?(?:(?:[ \t]*-[ \t]*|[ \t]*,[ \t]*|[ \t]+)([0-9]{4}|[0-9]{2})(?![0-9]))?)",std::regex::icase);
        static const std::regex numeric_date(R"(([0-9]{1,4})[ \t]*([/-])[ \t]*([0-9]{1,2})[ \t]*\2[ \t]*([0-9]{1,4})(?![0-9]))");
        static const std::regex time(R"(([0-9]{1,2})[ \t]*:[ \t]*([0-9]{2})(?:[ \t]*:[ \t]*([0-9]{2}))?(?:[ \t]*([ap])[ \t]*\.?[ \t]*m\.?)?)",std::regex::icase);
        static const char* month_names[] = {"January","February","March","April","May","June","July","August","September","October","November","December"};
        static const char* day_names[] = {"Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"};
        auto lower = [](std::string value) {
            for (char& c : value) if (c >= 'A' && c <= 'Z') c += 'a'-'A';
            return value;
        };
        auto expand_year = [](const std::string& value) {
            int y = std::stoi(value);
            return value.size() == 2 ? y + (y < 70 ? 2000 : 1900) : y;
        };
        std::smatch m;
        bool named = false, reverse = false;
        if (!french && alpha(s[at])) named = match(month_first,m);
        if (!french && !named && (alpha(s[at]) || digit(s[at]))) {
            named = match(named_day_first,m); reverse = true;
        }
        if (named) {
            auto month = lower(m[reverse ? 3 : 2].str()).substr(0,3);
            int month_index = 0;
            while (month_index < 12 && lower(month_names[month_index]).substr(0,3) != month) ++month_index;
            int d = std::stoi(m[reverse ? 2 : 3]), y = m[4].matched ? expand_year(m[4]) : 2000;
            if (month_index < 12 && valid_date(month_index+1,d,y)) {
                spoken.clear();
                if (m[1].matched) {
                    auto day = lower(m[1]).substr(0,3);
                    for (const char* name : day_names) if (lower(name).substr(0,3) == day) spoken = std::string(name)+", ";
                }
                spoken += std::string(month_names[month_index])+" "+spell(std::to_string(d),"ordinal");
                if (m[4].matched) spoken += ", "+year_or_number(std::to_string(y));
                end=at+m.length(); return true;
            }
        }
        if (!digit(s[at])) return false;
        if (match(numeric_date,m)) {
            bool iso = m[1].length() == 4;
            std::string ys = m[iso ? 1 : 4], ms = m[iso ? 3 : 1], ds = m[iso ? 4 : 3];
            int y = expand_year(ys), month = std::stoi(ms), day = std::stoi(ds);
            if ((ys.size() == 2 || ys.size() == 4) && ms.size() <= 2 && ds.size() <= 2 && valid_date(month,day,y)) {
                // Retain numeric component order, but read the year colloquially.
                auto yr = year_or_number(std::to_string(y));
                spoken = iso ? yr+" "+number(unpad(ms))+" "+number(unpad(ds)) :
                    number(unpad(ms))+" "+number(unpad(ds))+" "+yr;
                end=at+m.length(); return true;
            }
        }
        if (match(time,m)) {
            int h=std::stoi(m[1]), minute=std::stoi(m[2]), second=m[3].matched ? std::stoi(m[3]) : 0;
            bool meridiem=m[4].matched;
            if (h > 23 || minute > 59 || second > 59 || (meridiem && (h < 1 || h > 12))) return false;
            spoken=number(unpad(m[1]));
            if (french) {
                spoken += h == 1 ? " heure" : " heures";
                if (minute) spoken += " "+number(unpad(m[2]));
                if (m[3].matched) spoken += " "+number(unpad(m[3]))+(second == 1 ? " seconde" : " secondes");
                if (meridiem) spoken += lower(m[4]) == "a" ? " du matin" : " de l'après-midi";
            } else {
                if (minute || m[3].matched) spoken += (minute < 10 ? " oh " : " ")+number(unpad(m[2]));
                else if (!meridiem) spoken += " o'clock";
                if (m[3].matched) spoken += " and "+number(unpad(m[3]))+(second == 1 ? " second" : " seconds");
                if (meridiem) spoken += lower(m[4]) == "a" ? " ay em" : " pee em";
            }
            end=at+m.length();
            if (s[end-1] == '.') {
                size_t next = s.find_first_not_of(" \t\r\n",end);
                // A final abbreviation dot can also end a sentence.
                if (next == std::string::npos || (s[next] >= 'A' && s[next] <= 'Z')) spoken += '.';
            }
            return true;
        }
        return false;
    }

    std::string process(const std::string& raw) const {
        if (!enabled) return raw;
        wide(raw); // reject malformed UTF-8 consistently, including legacy input
        std::string s = substitutions(raw);
        replace_all(s, "\xC2\xA0", " "); replace_all(s, "\xE2\x80\xAF", " ");
        replace_all(s, "−", "-");
        static const std::regex phone(R"((?:\+[0-9]{1,3}[ .-]?(?:\([0-9]{1,4}\)|[0-9]{1,4})(?:[ .-][0-9]{2,4}){2,4}|\([0-9]{3}\)[ .-]?[0-9]{3}[ .-][0-9]{4}|[0-9]{3}[ .-][0-9]{3}[ .-][0-9]{4}|[0-9]{3}-[0-9]{4}|0[0-9](?:[ .-][0-9]{2}){4})(?:[ \t]*(?:ext\.?|extension|poste|x)[ \t]*[0-9]+)?)", std::regex::icase);
        static const std::regex ext(R"([ \t]*(?:ext\.?|extension|poste|x)[ \t]*([0-9]+)$)", std::regex::icase);
        static const std::regex token(R"([A-Za-z0-9]+(?:[-_][A-Za-z0-9]+|\([A-Za-z0-9_-]+\))*)");
        static const std::regex parens(R"(\([0-9]+\))");
        static const std::regex degree(R"([ \t]*(?:°|º|o)(C|F|K))");
        static const std::regex ordinal_en(R"((st|nd|rd|th))", std::regex::icase);
        static const std::regex ordinal_fr(R"((er|re|ème|e))");
        static const std::regex range(R"(([0-9]+)[ \t]*-[ \t]*([0-9]+))");
        static const std::regex separated(R"([A-Za-z0-9]+(?:[ \t]*[:/-][ \t]*[A-Za-z0-9]+)+)");
        static const std::regex pieces(R"([A-Za-z]+|[0-9]+)");
        static const std::regex month_piece(R"((?:^|[^A-Za-z])(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:$|[^A-Za-z]))",std::regex::icase);
        static const std::regex numeric_code(R"([0-9]+(?:[ \t]*-[ \t]*[0-9]+){2,})");
        // Capture the mantissa and exponent without evaluating a floating-point
        // value. Exponents use cardinal readings, never the bare-year heuristic.
        static const std::regex scientific_en(R"(([+-]?[0-9]+(?:\.[0-9]+)?)(?:[eE]|[ \t]*(?:×|x)[ \t]*10[ \t]*\^[ \t]*)([+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+(?![0-9])|[0-9]+)))");
        static const std::regex scientific_fr(R"(([+-]?[0-9]+(?:[.,][0-9]+)?)(?:[eE]|[ \t]*(?:×|x)[ \t]*10[ \t]*\^[ \t]*)([+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+(?![0-9])|[0-9]+)))");
        std::string out;
        for (size_t i = 0; i < s.size();) {
            auto match = [&](size_t at, const std::regex& re, std::smatch& m) {
                return std::regex_search(s.cbegin()+at, s.cend(), m, re, std::regex_constants::match_continuous);
            };
            // Keep URLs and email addresses intact rather than rewriting their
            // digits/units. They need a separate, explicitly designed policy.
            if (!word_before(s,i) && s[i] != ' ') {
                size_t end = s.find_first_of(" \t\r\n", i);
                if (end == std::string::npos) end = s.size();
                auto piece = s.substr(i, end-i);
                if (piece.find('@') != std::string::npos || piece.find("://") != std::string::npos || piece.compare(0,4,"www.") == 0) {
                    out += piece; i = end; continue;
                }
            }
            if (word_before(s,i)) { out += s[i++]; continue; }
            size_t contextual_end;
            std::string contextual;
            if (date_time(s,i,contextual_end,contextual)) {
                out += contextual; i=contextual_end; continue;
            }
            std::smatch m, n;
            if ((digit(s[i]) || s[i] == '+' || s[i] == '-') &&
                match(i,french ? scientific_fr : scientific_en,m) && boundary(s,i+m.length())) {
                size_t end = i+m.length();
                // Do not consume a prefix of a structured identifier or a
                // malformed fractional/grouped exponent.
                bool continuation = end < s.size() && (s[end] == '-' || s[end] == '_' ||
                    ((s[end] == '.' || s[end] == ',') && end+1 < s.size() && digit(s[end+1])));
                if (!continuation) {
                    auto mantissa = m[1].str(), exponent = m[2].str();
                    replace_all(mantissa,",","."); replace_all(exponent,",","");
                    bool negative = exponent[0] == '-';
                    if (negative || exponent[0] == '+') exponent.erase(0,1);
                    auto first = exponent.find_first_not_of('0');
                    exponent = first == std::string::npos ? "0" : exponent.substr(first);
                    out += number(mantissa) + (french ? " fois dix à la puissance " : " times ten to the ") +
                        (negative ? (french ? "moins " : "minus ") : "") + number(exponent);
                    i=end; continue;
                }
            }
            if ((digit(s[i]) || s[i] == '(' || s[i] == '+') && match(i, phone, m) && boundary(s, i+m.length()) &&
                (i+m.length() == s.size() || (s[i+m.length()] != '-' && s[i+m.length()] != '_'))) {
                auto value = m.str();
                std::smatch extension;
                bool has_ext = std::regex_search(value, extension, ext);
                out += (value[0] == '+' ? "plus " : "") + phone_digits(has_ext ? value.substr(0, extension.position()) : value);
                if (has_ext) out += (french ? ", poste " : ", extension ") + digits(extension[1]);
                i += m.length(); continue;
            }
            // Recognized phones have priority. A chain of three or more groups
            // is an identifier; do not reinterpret a trailing pair as a range.
            if (digit(s[i]) && match(i,numeric_code,m) && boundary(s,i+m.length())) {
                size_t after = i+m.length();
                while (after < s.size() && (s[after] == ' ' || s[after] == '\t')) ++after;
                if (after == s.size() || s[after] != '-') {
                    out += digits(m.str(),true); i+=m.length(); continue;
                }
            }
            if (digit(s[i]) && match(i,range,m) && boundary(s,i+m.length())) {
                size_t after = i+m.length();
                while (after < s.size() && (s[after] == ' ' || s[after] == '\t')) ++after;
                if (after == s.size() || s[after] != '-') {
                    out += year_or_number(m[1]) + (french ? " à " : " to ") + year_or_number(m[2]);
                    i+=m.length(); continue;
                }
            }
            // Unrecognized date/time-like forms: retain alphabetic runs and
            // speak numeric runs as groups, dropping their separators. Explicit
            // phone, range and structured identifier policies still take priority.
            if ((digit(s[i]) || alpha(s[i])) && match(i,separated,m) && boundary(s,i+m.length())) {
                auto value = m.str();
                if (std::any_of(value.begin(),value.end(),digit) &&
                    (value.find_first_of(":/") != std::string::npos || std::regex_search(value,month_piece))) {
                    std::string fallback;
                    for (std::sregex_iterator it(value.begin(),value.end(),pieces), last; it != last; ++it) {
                        auto piece = it->str();
                        append(fallback,digit(piece[0]) ? year_or_number(unpad(piece)) : piece);
                    }
                    out += fallback; i+=m.length(); continue;
                }
            }
            // Prefix currency symbols can be UTF-8; optional sign before them.
            size_t cp = i;
            std::string sign;
            if (s[cp] == '-' || s[cp] == '+') { sign = s.substr(cp,1); ++cp; }
            std::string symbol;
            for (auto sym : {"$", "€", "£"}) if (s.compare(cp, std::char_traits<char>::length(sym), sym) == 0) { symbol = sym; break; }
            if (!symbol.empty()) {
                cp += symbol.size(); while (cp < s.size() && s[cp] == ' ') ++cp;
                if (match(cp, numeric, m)) {
                    size_t end = cp + m.length(); auto scale = scale_at(s,end);
                    if (boundary(s,end)) { out += money(sign + canonical(m.str()),symbol,scale); i = end; continue; }
                }
            }
            if (match(i, numeric, m)) {
                size_t end = i + m.length(), suffix = end;
                auto value = canonical(m.str());
                while (suffix < s.size() && (s[suffix] == ' ' || s[suffix] == '\t')) ++suffix;
                // A hyphen can join a quantity to its unit: 150-mg, 12-h.
                if (suffix == end && suffix < s.size() && s[suffix] == '-') ++suffix;
                bool consumed = false;
                size_t magnitude_end = end;
                auto magnitude = scale_at(s,magnitude_end);
                size_t currency_at = magnitude.empty() ? suffix : magnitude_end;
                while (currency_at < s.size() && (s[currency_at] == ' ' || s[currency_at] == '\t')) ++currency_at;
                for (auto sym : {"$", "€", "£"}) {
                    auto len = std::char_traits<char>::length(sym);
                    if (s.compare(currency_at,len,sym) == 0 && boundary(s,currency_at+len)) {
                        out += money(value,sym,magnitude); i=currency_at+len; consumed=true; break;
                    }
                }
                if (consumed) continue;
                if (match(end, degree, n) && boundary(s,end+n.length())) {
                    std::string scale = n[1] == "C" ? "Celsius" : n[1] == "F" ? "Fahrenheit" : "Kelvin";
                    out += number(value) + (n[1] == "K" ? " " : french ? (singular(value) ? " degré " : " degrés ") : (singular(value) ? " degree " : " degrees ")) + scale;
                    i=end+n.length(); continue;
                }
                for (const auto& unit : units) {
                    size_t len = std::char_traits<char>::length(unit.suffix);
                    if (s.compare(suffix,len,unit.suffix) == 0 && boundary(s,suffix+len)) {
                        out += number(value,unit.feminine) + " " + (french ? (singular(value) ? unit.fr : unit.frs) : (singular(value) ? unit.en : unit.ens));
                        i=suffix+len; consumed=true; break;
                    }
                }
                if (consumed) continue;
                if (value.find('.') == std::string::npos && match(end,french ? ordinal_fr : ordinal_en,n) && boundary(s,end+n.length())) {
                    // Only accept consistent English suffixes (1st, 12th, etc.).
                    std::string tail=n.str();
                    std::transform(tail.begin(),tail.end(),tail.begin(),[](char c){ return c>='A'&&c<='Z' ? c+32 : c; });
                    auto digits_value=value; if (digits_value[0]=='+' || digits_value[0]=='-') digits_value.erase(0,1);
                    int last=digits_value.back()-'0';
                    bool teen=digits_value.size()>1 && digits_value[digits_value.size()-2]=='1';
                    std::string expected=teen ? "th" : last==1 ? "st" : last==2 ? "nd" : last==3 ? "rd" : "th";
                    if (french || tail==expected) { out += spell(value, french && tail=="re" ? "ordinal-feminine" : "ordinal"); i=end+n.length(); continue; }
                }
                if (end < s.size() && s[end] == '-' && match(end+1,token,n)) {
                    auto tail = n.str();
                    if (std::any_of(tail.begin(),tail.end(),[](char c) { return c >= 'a' && c <= 'z'; })) {
                        out += number(value) + " " + read_token(tail,true);
                        i=end+1+n.length(); continue;
                    }
                }
                // A mixed identifier must win over the ordinary numeric prefix.
                if (end == s.size() || (!alpha(s[end]) && s[end] != '-' && s[end] != '_' && s[end] != '(')) {
                    out += m.str() == value ? year_or_number(value) : number(value); i=end; continue;
                }
            }
            if (match(i, parens, m)) { out += digits(m.str()); i+=m.length(); continue; }
            if (match(i,token,m)) {
                auto value=m.str();
                out += read_token(value);
                i+=m.length(); continue;
            }
            out += s[i++];
        }
        return out;
    }
};

Normalizer::Normalizer(const std::string& language, const std::string& path, bool enabled)
    : impl_(new Impl(language,path,enabled)) {}
Normalizer::~Normalizer() = default;
std::string Normalizer::process(const std::string& text) const { return impl_->process(text); }
}
