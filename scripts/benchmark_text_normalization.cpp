// Standalone latency benchmark, including default CSV substitutions. No models.
#include "../native_text/normalizer.h"
#include "../json.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
        int repeats = argc > 1 ? std::stoi(argv[1]) : 1000;
        if (repeats < 1 || repeats > 1000000) throw std::runtime_error("repeats must be 1..1000000");
        using Clock = std::chrono::steady_clock;
        auto ms = [](Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); };
        struct Case { std::string name, language, text; };
        std::string paragraph;
        while (paragraph.size() < 440) paragraph += "Dr. Smith paid $14 million for 150-milligrams of material. The 3-D-printer produces 12-34 items. ";
        paragraph.resize(440);
        std::vector<Case> cases = {
            {"prose", "english", "The morning sun lit the quiet street. A narrator paused before telling the next part of the story."},
            {"currency", "english", "The startup secured $14 million in venture capital, a huge leap from their initial $450K seed round."},
            {"ranges_units", "english", "A 3-D-printer uses 150-milligrams of material to make 12-34 items. Wait 30s at 78oC."},
            {"phone_codes", "english", "Call (212) 555-0142 ext. 402. The code is 554-33-2998 or 87M-34KEL6."},
            {"dates_times", "english", "Meet Mon. Jan. 1, 1980 at 11 : 06 P. M. The next date is 12 - Mar - 09 or 2026-03-12."},
            {"fiscal_year", "english", "FY2026 results exceeded FY2025 expectations. The budget is $14 million."},
            {"paragraph_440", "english", paragraph},
            {"french", "french", "M. Dupont a payé 14 millions € pour 12-34 articles. Le code est 554-33-2998. Il fait 28°C."},
        };
        nlohmann::json report = {{"repeats",repeats}, {"note","One process. First numeric conversion in each language includes lazy rule initialization. Times are milliseconds; constructor includes CSV loading."}, {"cases",nlohmann::json::array()}};
        for (const auto& c : cases) {
            auto start = Clock::now();
            pocket_speech::Normalizer n(c.language);
            double constructor = ms(start);
            start = Clock::now(); auto expected = n.process(c.text); double first = ms(start);
            std::vector<double> times; times.reserve(repeats);
            for (int i = 0; i < repeats; ++i) {
                start = Clock::now(); auto value = n.process(c.text); times.push_back(ms(start));
                if (value != expected) throw std::runtime_error("Nondeterministic normalization");
            }
            std::sort(times.begin(),times.end());
            report["cases"].push_back({{"case",c.name},{"language",c.language},{"input_bytes",c.text.size()},
                {"constructor_ms",constructor},{"first_call_ms",first},
                {"warm_mean_ms",std::accumulate(times.begin(),times.end(),0.0)/repeats},
                {"warm_median_ms",times[repeats/2]},{"warm_p95_ms",times[std::min(repeats-1,int(repeats*0.95))]},
                {"normalized_text",expected}});
        }
        std::cout << report.dump(2) << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
