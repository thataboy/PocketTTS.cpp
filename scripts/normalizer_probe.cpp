// JSON-lines probe for text policy tests; no model/runtime dependencies.
#include "../native_text/normalizer.h"
#include "../json.hpp"
#include <iostream>

int main() {
    std::string line;
    while (std::getline(std::cin,line)) {
        nlohmann::json result;
        try {
            auto q=nlohmann::json::parse(line);
            pocket_speech::Normalizer n(q.value("language","english"), q.value("lexicon","-"), q.value("enabled",true));
            result={{"text",n.process(q.at("text"))}};
        } catch (const std::exception& e) { result={{"error",e.what()}}; }
        std::cout << result.dump() << '\n';
    }
}
