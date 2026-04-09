// ---------------- iTalk state ----------------
std::mutex italk_mutex_;
std::filesystem::path italk_data_path_ = []() {
    const char* home = std::getenv("HOME");
    return home ? std::filesystem::path(home) : std::filesystem::current_path();
}() / "italk.json";

// ---------------- Helpers ----------------
static std::string trim_copy(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

static std::string normalize_spaces_lower(std::string s) {
    std::string out;
    out.reserve(s.size());
    bool in_space = false;
    for (unsigned char ch : s) {
        if (std::isspace(ch)) {
            if (!in_space && !out.empty()) out.push_back(' ');
            in_space = true;
        } else {
            out.push_back(static_cast<char>(std::tolower(ch)));
            in_space = false;
        }
    }
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

static std::string random_hex(size_t n) {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    static const char* hex = "0123456789abcdef";
    std::string s;
    s.reserve(n);
    for (size_t i = 0; i < n; ++i) s.push_back(hex[rng() & 0xF]);
    return s;
}

static std::string italk_now_iso() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t tt = system_clock::to_time_t(now);

    std::tm local_tm{};
#ifdef _WIN32
    localtime_s(&local_tm, &tt);
#else
    localtime_r(&tt, &local_tm);
#endif

    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S%z", &local_tm);

    std::string s(buf);
    if (s.size() >= 5) {
        s.insert(s.size() - 2, ":");
    }
    return s;
}

static json make_tag(
    const std::string& id,
    const std::string& label,
    const std::string& text,
    const std::string& created_at
) {
    return {
        {"id", id},
        {"label", label},
        {"text", text},
        {"created_at", created_at}
    };
}

static json make_favorite(
    const std::string& id,
    const std::string& text,
    const std::string& created_at,
    const std::string& updated_at
) {
    return {
        {"id", id},
        {"text", text},
        {"created_at", created_at},
        {"updated_at", updated_at}
    };
}

static json make_session_line(
    const std::string& id,
    const std::string& text,
    const std::string& ts
) {
    return {
        {"id", id},
        {"text", text},
        {"ts", ts}
    };
}

json italk_default_state() {
    const std::string now = italk_now_iso();

    json root = json::object();
    root["meta"]["version"] = 1;
    root["meta"]["created_at"] = now;

    root["settings"]["last_voice"] = "";
    root["settings"]["last_tab"] = "";
    root["settings"]["expanded_fav_categories"] = json::array();

    root["tags"] = json::array({
        make_tag("tag_yes", "Yes", "Yes", now),
        make_tag("tag_no", "No", "No", now),
        make_tag("tag_repeat", "Repeat", "Could you repeat that", now),
        make_tag("tag_xcus", "Xcus", "Excuse me", now),
        make_tag("tag_thx", "Thx", "Thank you", now),
        make_tag("tag_tkvm", "TKVM", "Thank you very much", now),
        make_tag("tag_hand", "HAND", "Have a nice day", now),
        make_tag("tag_bye", "Bye", "Bye", now)
    });

    root["favorites"]["categories"]["Intro"] = json::array();
    root["favorites"]["categories"]["Contact"] = json::array();
    root["favorites"]["categories"]["Insurance"] = json::array();

    root["last_session"]["name"] = "";
    root["last_session"]["started_at"] = now;
    root["last_session"]["lines"] = json::array();

    root["history"] = json::array();
    return root;
}

json italk_load() {
    json data;
    try {
        if (std::filesystem::exists(italk_data_path_)) {
            std::ifstream f(italk_data_path_);
            f >> data;
        }
    } catch (...) {
        data = nullptr;
    }

    if (data.is_null() || !data.is_object()) {
        return italk_default_state();
    }

    json def = italk_default_state();

    // Structural validation / migration
    if (!data.contains("meta") || !data["meta"].is_object()) data["meta"] = def["meta"];
    if (!data.contains("settings") || !data["settings"].is_object()) data["settings"] = def["settings"];
    if (!data["settings"].contains("last_voice")) data["settings"]["last_voice"] = "";
    if (!data["settings"].contains("last_tab")) data["settings"]["last_tab"] = "";
    if (!data["settings"].contains("expanded_fav_categories") || !data["settings"]["expanded_fav_categories"].is_array()) {
        data["settings"]["expanded_fav_categories"] = json::array();
    }

    if (!data.contains("tags") || !data["tags"].is_array()) data["tags"] = def["tags"];

    if (!data.contains("favorites") || !data["favorites"].is_object()) data["favorites"] = def["favorites"];
    if (!data["favorites"].contains("categories") || !data["favorites"]["categories"].is_object()) {
        data["favorites"]["categories"] = def["favorites"]["categories"];
    }

    if (!data.contains("last_session") || !data["last_session"].is_object()) data["last_session"] = def["last_session"];
    if (!data["last_session"].contains("name")) data["last_session"]["name"] = "";
    if (!data["last_session"].contains("started_at")) data["last_session"]["started_at"] = italk_now_iso();
    if (!data["last_session"].contains("lines") || !data["last_session"]["lines"].is_array()) {
        data["last_session"]["lines"] = json::array();
    }

    if (!data.contains("history") || !data["history"].is_array()) data["history"] = json::array();

    return data;
}

void italk_save(const json& data) {
    auto tmp = italk_data_path_;
    tmp += ".tmp";

    {
        std::ofstream f(tmp);
        // Indent 4 spaces
        f << data.dump(4);
    }

    std::filesystem::rename(tmp, italk_data_path_);
}

static std::pair<std::string, json*> italk_find_fav(json& data, const std::string& fav_id) {
    auto& cats = data["favorites"]["categories"];

    for (auto& [cat_name, items] : cats.items()) {
        if (!items.is_array()) continue;
        for (auto& fav : items) {
            if (fav.contains("id") && fav["id"] == fav_id) {
                return {cat_name, &fav};
            }
        }
    }
    return {"", nullptr};
}

static bool erase_by_id(json& items, const std::string& id) {
    auto it = std::find_if(items.begin(), items.end(), [&](const json& j) {
        return j.contains("id") && j["id"] == id;
    });
    if (it != items.end()) {
        items.erase(it);
        return true;
    }
    return false;
}

static bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

static std::vector<std::string> split_path(const std::string& path) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : path) {
        if (c == '/') {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

bool handle_italk_request(ptt_socket_t client_fd, const HttpRequest& req) {
    if (!starts_with(req.path, "/italk")) return false;

    const std::string subpath = req.path.substr(std::string("/italk").size()).empty()
        ? "/"
        : req.path.substr(std::string("/italk").size());

    json body_json;
    if (req.method == "POST" || req.method == "PUT") {
        try {
            if (!req.body.empty()) body_json = json::parse(req.body);
        } catch (const std::exception& e) {
            return send_json_error(client_fd, 400, std::string("Invalid JSON: ") + e.what()), true;
        }
    }

    // GET /italk/state
    if (req.method == "GET" && subpath == "/state") {
        std::lock_guard<std::mutex> lock(italk_mutex_);
        json data = italk_load();
        return send_response(client_fd, 200, "application/json", data.dump()), true;
    }

    // POST /italk/settings/voice
    if (req.method == "POST" && subpath == "/settings/voice") {
        std::string voice = trim_copy(body_json.value("voice", ""));
        std::lock_guard<std::mutex> lock(italk_mutex_);
        json data = italk_load();
        data["settings"]["last_voice"] = voice;
        italk_save(data);
        return send_json_ok(client_fd), true;
    }

    // POST /italk/settings/tab
    if (req.method == "POST" && subpath == "/settings/tab") {
        std::string tab = trim_copy(body_json.value("tab", ""));
        std::lock_guard<std::mutex> lock(italk_mutex_);
        json data = italk_load();
        data["settings"]["last_tab"] = tab;
        italk_save(data);
        return send_json_ok(client_fd), true;
    }

    // POST /italk/settings/fav_category
    if (req.method == "POST" && subpath == "/settings/fav_category") {
        std::string cat = trim_copy(body_json.value("category", ""));
        std::lock_guard<std::mutex> lock(italk_mutex_);
        json data = italk_load();
        data["settings"]["last_fav_category"] = cat;
        italk_save(data);
        return send_json_ok(client_fd), true;
    }

    // POST /italk/settings/expanded_categories
    if (req.method == "POST" && subpath == "/settings/expanded_categories") {
        if (!body_json.contains("categories") || !body_json["categories"].is_array()) {
            return send_json_error(client_fd, 400, "categories required"), true;
        }
        std::lock_guard<std::mutex> lock(italk_mutex_);
        json data = italk_load();
        data["settings"]["expanded_fav_categories"] = body_json["categories"];
        italk_save(data);
        return send_json_ok(client_fd), true;
    }

    auto send_id = [&](const std::string& id) {
        json j;
        j["id"] = id;
        return send_response(client_fd, 200, "application/json", j.dump());
    };

    // POST /italk/tags
    if (req.method == "POST" && subpath == "/tags") {
        std::string label = trim_copy(body_json.value("label", ""));
        std::string text = trim_copy(body_json.value("text", ""));
        if (text.empty()) return send_json_error(client_fd, 400, "text required"), true;

        std::string tag_id = "tag_" + random_hex(8);

        std::lock_guard<std::mutex> lock(italk_mutex_);
        json data = italk_load();
        data["tags"].push_back(make_tag(
            tag_id,
            !label.empty() ? label : text.substr(0, std::min<size_t>(12, text.size())),
            text,
            italk_now_iso()
        ));
        italk_save(data);
        return send_id(tag_id), true;
    }

    // PUT /italk/tags/{tag_id}
    if (req.method == "PUT" && starts_with(subpath, "/tags/")) {
        auto parts = split_path(subpath);
        if (parts.size() == 2 && parts[0] == "tags") {
            std::string tag_id = parts[1];
            std::string label = trim_copy(body_json.value("label", ""));
            std::string text = trim_copy(body_json.value("text", ""));

            std::lock_guard<std::mutex> lock(italk_mutex_);
            json data = italk_load();
            for (auto& t : data["tags"]) {
                if (t["id"] == tag_id) {
                    if (!label.empty()) t["label"] = label;
                    if (!text.empty()) t["text"] = text;
                }
            }
            italk_save(data);
            return send_json_ok(client_fd), true;
        }
    }

    // DELETE /italk/tags/{tag_id}
    if (req.method == "DELETE" && starts_with(subpath, "/tags/")) {
        auto parts = split_path(subpath);
        if (parts.size() == 2 && parts[0] == "tags") {
            std::string tag_id = parts[1];

            std::lock_guard<std::mutex> lock(italk_mutex_);
            json data = italk_load();
            if (erase_by_id(data["tags"], tag_id))
                italk_save(data);
            return send_json_ok(client_fd), true;
        }
    }

    // POST /italk/tags/reorder
    if (req.method == "POST" && subpath == "/tags/reorder") {
        if (!body_json.contains("ids") || !body_json["ids"].is_array()) {
            return send_json_error(client_fd, 400, "ids required"), true;
        }

        std::lock_guard<std::mutex> lock(italk_mutex_);
        json data = italk_load();
        std::unordered_map<std::string, json> tag_map;
        for (auto& t : data["tags"]) tag_map[t["id"].get<std::string>()] = t;

        json reordered = json::array();
        for (const auto& id_val : body_json["ids"]) {
            std::string tid = id_val.get<std::string>();
            if (tag_map.count(tid)) reordered.push_back(tag_map[tid]);
        }

        data["tags"] = reordered;
        italk_save(data);
        return send_json_ok(client_fd), true;
    }

    // POST /italk/favorites
    if (req.method == "POST" && subpath == "/favorites") {
        std::string category = trim_copy(body_json.value("category", "Unsorted"));
        if (category.empty()) category = "Unsorted";
        std::string text = trim_copy(body_json.value("text", ""));
        if (text.empty()) return send_json_error(client_fd, 400, "text required"), true;

        std::string fav_id = "fav_" + random_hex(8);
        const std::string now = italk_now_iso();

        std::lock_guard<std::mutex> lock(italk_mutex_);
        json data = italk_load();
        auto& cats = data["favorites"]["categories"];
        if (!cats.contains(category) || !cats[category].is_array()) {
            cats[category] = json::array();
        }

        cats[category].push_back(make_favorite(fav_id, text, now, now));
        italk_save(data);
        return send_id(fav_id), true;
    }

    // PUT /italk/favorites/{fav_id}
    if (req.method == "PUT" && starts_with(subpath, "/favorites/")) {
        auto parts = split_path(subpath);
        if (parts.size() == 2 && parts[0] == "favorites") {
            std::string fav_id = parts[1];
            std::string category = trim_copy(body_json.value("category", ""));
            if (category.empty()) return send_json_error(client_fd, 404, "category empty"), true;
            std::string text = trim_copy(body_json.value("text", ""));

            std::lock_guard<std::mutex> lock(italk_mutex_);
            json data = italk_load();

            auto [_, fav] = italk_find_fav(data, fav_id);
            if (fav == nullptr) return send_json_error(client_fd, 404, "favorite not found"), true;
            (*fav)["text"] = text;
            (*fav)["updated_at"] = italk_now_iso();
            italk_save(data);
            return send_json_ok(client_fd), true;
        }
    }

    // DELETE /italk/favorites/{fav_id}
    if (req.method == "DELETE" && starts_with(subpath, "/favorites/")) {
        auto parts = split_path(subpath);
        if (parts.size() == 2 && parts[0] == "favorites") {
            std::string fav_id = parts[1];
            std::lock_guard<std::mutex> lock(italk_mutex_);
            json data = italk_load();
            auto [cat, _] = italk_find_fav(data, fav_id);
            if (!cat.empty() && erase_by_id(data["favorites"]["categories"][cat], fav_id))
                italk_save(data);
            return send_json_ok(client_fd), true;
        }
    }

    // POST /italk/favorites/reorder
    if (req.method == "POST" && subpath == "/favorites/reorder") {
        std::string from_cat = body_json.value("from_cat", "");
        std::string to_cat = body_json.value("to_cat", "");
        int from_idx = body_json.value("from", -1);
        int to_idx = body_json.value("to", -1);

        std::lock_guard<std::mutex> lock(italk_mutex_);
        json data = italk_load();
        auto& cats = data["favorites"]["categories"];

        if (!cats.contains(from_cat) || !cats.contains(to_cat)) {
            return send_json_error(client_fd, 400, "Category not found"), true;
        }

        auto& items_from = cats[from_cat];
        if (from_idx < 0 || static_cast<size_t>(from_idx) >= items_from.size()) {
            return send_json_error(client_fd, 400, "Index out of range"), true;
        }

        json item = items_from[from_idx];
        items_from.erase(items_from.begin() + from_idx);

        auto& items_to = cats[to_cat];
        int target = std::max(0, std::min<int>(to_idx, static_cast<int>(items_to.size())));
        items_to.insert(items_to.begin() + target, item);

        italk_save(data);
        return send_json_ok(client_fd), true;
    }

    // POST /italk/session/append_line
    if (req.method == "POST" && subpath == "/session/append_line") {
        std::string text = trim_copy(body_json.value("text", ""));
        if (text.empty()) return send_json_ok(client_fd), true;

        std::string line_id = "line_" + random_hex(10);
        std::string norm = normalize_spaces_lower(text);

        std::lock_guard<std::mutex> lock(italk_mutex_);
        json data = italk_load();
        auto& lines = data["last_session"]["lines"];
        auto it = std::find_if(lines.begin(), lines.end(), [&](const json& j) {
            return normalize_spaces_lower(j.value("text", "")) == norm;
        });
        if (it != lines.end()) lines.erase(it);
        lines.insert(lines.begin(), make_session_line(line_id, text, italk_now_iso()));
        italk_save(data);
        return send_id(line_id), true;
    }

    // POST /italk/session/delete_line
    if (req.method == "POST" && subpath == "/session/delete_line") {
        std::string line_id = trim_copy(body_json.value("id", ""));
        if (line_id.empty()) return send_json_ok(client_fd), true;

        std::lock_guard<std::mutex> lock(italk_mutex_);
        json data = italk_load();
        if (erase_by_id(data["last_session"]["lines"], line_id))
            italk_save(data);
        return send_json_ok(client_fd), true;
    }

    // POST /italk/history/delete_line
    if (req.method == "POST" && subpath == "/history/delete_line") {
        std::string session_id = trim_copy(body_json.value("session_id", ""));
        std::string line_id = trim_copy(body_json.value("line_id", ""));
        if (session_id.empty() || line_id.empty()) return send_json_ok(client_fd), true;

        std::lock_guard<std::mutex> lock(italk_mutex_);
        json data = italk_load();

        for (auto& sess : data["history"]) {
            if (sess["id"] == session_id) {
                if (erase_by_id(sess["lines"], line_id))
                    italk_save(data);
                break;
            }
        }

        return send_json_ok(client_fd), true;
    }

    // POST /italk/session/name
    if (req.method == "POST" && subpath == "/session/name") {
        std::string name = trim_copy(body_json.value("name", ""));
        std::lock_guard<std::mutex> lock(italk_mutex_);
        json data = italk_load();
        data["last_session"]["name"] = name;
        italk_save(data);
        return send_json_ok(client_fd), true;
    }

    // POST /italk/session/new | /session/clear
    if (req.method == "POST" && (subpath == "/session/new" || subpath == "/session/clear")) {
        std::lock_guard<std::mutex> lock(italk_mutex_);
        json data = italk_load();
        data["last_session"]["name"] = "";
        data["last_session"]["started_at"] = italk_now_iso();
        data["last_session"]["lines"] = json::array();
        italk_save(data);
        return send_json_ok(client_fd), true;
    }

    // POST /italk/session/save
    if (req.method == "POST" && subpath == "/session/save") {
        std::string name = trim_copy(body_json.value("name", "Untitled"));
        std::string sess_id;

        std::lock_guard<std::mutex> lock(italk_mutex_);
        json data = italk_load();
        auto& last = data["last_session"];

        last["name"] = name;

        if (last["lines"].is_array() && !last["lines"].empty()) {
            sess_id = "sess_" + random_hex(8);
            json hist_item = {
                {"id", sess_id},
                {"name", last.value("name", "")},
                {"saved_at", italk_now_iso()},
                {"lines", last["lines"]}
            };

            data["history"].insert(data["history"].begin(), hist_item);
            italk_save(data);
        }
        return send_id(sess_id), true;
    }

    // POST /italk/history/delete
    if (req.method == "POST" && subpath == "/history/delete") {
        std::string history_id = trim_copy(body_json.value("id", ""));
        std::lock_guard<std::mutex> lock(italk_mutex_);
        json data = italk_load();
        if (erase_by_id(data["history"], history_id))
            italk_save(data);
        return send_json_ok(client_fd), true;
    }

    return false;
}