#pragma once
#include <memory>
#include <string>

namespace pocket_speech {
// Immutable after construction; one instance per engine. No model dependency.
class Normalizer {
public:
    // Empty lexicon_path: look for lexicon.<language>.csv in the working directory.
    // "-": no lexicon. Explicit missing/malformed files are errors.
    Normalizer(const std::string& language, const std::string& lexicon_path = "", bool enabled = true);
    ~Normalizer();
    std::string process(const std::string& text) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
