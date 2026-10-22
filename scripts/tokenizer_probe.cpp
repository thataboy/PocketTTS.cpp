// Standalone JSON-lines probe for Python reference parity checks.
#include "../native_tokenizer/pocket_text.cpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        pocket_text::Tokenizer tok(argv[1]);
        std::string line;
        while (std::getline(std::cin, line)) {
            pocket_text::Json result;
            try {
                auto request = pocket_text::Json::parse(line);
                auto action = request.at("action").get<std::string>();
                if (action == "encode") result = tok.encode(request.at("text"));
                else if (action == "decode") result = tok.decode(request.at("ids").get<std::vector<int>>());
                else result = pocket_text::process(tok, request.at("text"), request.at("config"));
            } catch (const std::exception& error) { result = {{"error", error.what()}}; }
            std::cout << result.dump() << '\n';
        }
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
