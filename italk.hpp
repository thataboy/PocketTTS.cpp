// ---------------- iTalk state ----------------
std::mutex italk_mutex_;
std::filesystem::path italk_data_path_ = std::filesystem::current_path() / "italk.yaml";

// ---------------- Helpers ----------------
bool send_json_ok(ptt_socket_t fd) {
    json j;
    j["ok"] = true;
    return send_response(fd, 200, "application/json", j.dump());
}

bool send_json_error(ptt_socket_t fd, int status, const std::string& msg) {
    json j;
    j["error"] = msg;
    return send_response(fd, status, "application/json", j.dump());
}


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
        // Convert -0700 -> -07:00
        s.insert(s.size() - 2, ":");
    }
    return s;
}

static YAML::Node make_tag(
    const std::string& id,
    const std::string& label,
    const std::string& text,
    const std::string& created_at
) {
    YAML::Node n;
    n["id"] = id;
    n["label"] = label;
    n["text"] = text;
    n["created_at"] = created_at;
    return n;
}

static YAML::Node make_favorite(
    const std::string& id,
    const std::string& text,
    const std::string& created_at,
    const std::string& updated_at
) {
    YAML::Node n;
    n["id"] = id;
    n["text"] = text;
    n["created_at"] = created_at;
    n["updated_at"] = updated_at;
    return n;
}

static YAML::Node make_session_line(
    const std::string& id,
    const std::string& text,
    const std::string& ts
) {
    YAML::Node n;
    n["id"] = id;
    n["text"] = text;
    n["ts"] = ts;
    return n;
}

YAML::Node italk_default_state() const {
    const std::string now = italk_now_iso();

    YAML::Node root;
    root["meta"]["version"] = 1;
    root["meta"]["created_at"] = now;

    root["settings"]["last_voice"] = "";
    root["settings"]["expanded_fav_categories"] = YAML::Node(YAML::NodeType::Sequence);

    root["tags"] = YAML::Node(YAML::NodeType::Sequence);
    root["tags"].push_back(make_tag("tag_yes", "Yes", "Yes", now));
    root["tags"].push_back(make_tag("tag_no", "No", "No", now));
    root["tags"].push_back(make_tag("tag_repeat", "Repeat", "Could you repeat that", now));
    root["tags"].push_back(make_tag("tag_xcus", "Xcus", "Excuse me", now));
    root["tags"].push_back(make_tag("tag_thx", "Thx", "Thank you", now));
    root["tags"].push_back(make_tag("tag_tkvm", "TKVM", "Thank you very much", now));
    root["tags"].push_back(make_tag("tag_hand", "HAND", "Have a nice day", now));
    root["tags"].push_back(make_tag("tag_bye", "Bye", "Bye", now));

    root["favorites"]["categories"]["Intro"] = YAML::Node();
    root["favorites"]["categories"]["Contact"] = YAML::Node();
    root["favorites"]["categories"]["Insurance"] = YAML::Node();

    root["last_session"]["name"] = "";
    root["last_session"]["started_at"] = now;
    root["last_session"]["lines"] = YAML::Node(YAML::NodeType::Sequence);

    root["history"] = YAML::Node(YAML::NodeType::Sequence);
    return root;
}

YAML::Node ensure_map(YAML::Node parent, const std::string& key) const {
    if (!parent[key] || !parent[key].IsMap()) parent[key] = YAML::Node(YAML::NodeType::Map);
    return parent[key];
}

YAML::Node ensure_seq(YAML::Node parent, const std::string& key) const {
    if (!parent[key] || !parent[key].IsSequence()) parent[key] = YAML::Node(YAML::NodeType::Sequence);
    return parent[key];
}

YAML::Node italk_load() const {
    YAML::Node data;
    try {
        if (std::filesystem::exists(italk_data_path_)) {
            data = YAML::LoadFile(italk_data_path_.string());
        }
    } catch (...) {
        data = YAML::Node();
    }

    if (!data || !data.IsMap()) {
        return italk_default_state();
    }

    YAML::Node def = italk_default_state();

    if (!data["meta"] || !data["meta"].IsMap()) data["meta"] = def["meta"];
    if (!data["settings"] || !data["settings"].IsMap()) data["settings"] = def["settings"];
    if (!data["settings"]["last_voice"]) data["settings"]["last_voice"] = "";
    if (!data["settings"]["expanded_fav_categories"] || !data["settings"]["expanded_fav_categories"].IsSequence()) {
        data["settings"]["expanded_fav_categories"] = YAML::Node(YAML::NodeType::Sequence);
    }

    if (!data["tags"] || !data["tags"].IsSequence()) data["tags"] = def["tags"];

    if (!data["favorites"] || !data["favorites"].IsMap()) data["favorites"] = def["favorites"];
    if (!data["favorites"]["categories"] || !data["favorites"]["categories"].IsMap()) {
        data["favorites"]["categories"] = def["favorites"]["categories"];
    }

    if (!data["last_session"] || !data["last_session"].IsMap()) data["last_session"] = def["last_session"];
    if (!data["last_session"]["name"]) data["last_session"]["name"] = "";
    if (!data["last_session"]["started_at"]) data["last_session"]["started_at"] = italk_now_iso();
    if (!data["last_session"]["lines"] || !data["last_session"]["lines"].IsSequence()) {
        data["last_session"]["lines"] = YAML::Node(YAML::NodeType::Sequence);
    }

    if (!data["history"] || !data["history"].IsSequence()) data["history"] = YAML::Node(YAML::NodeType::Sequence);

    return data;
}

void italk_save(const YAML::Node& data) const {
    YAML::Emitter out;
    out << data;

    auto tmp = italk_data_path_;
    tmp += ".tmp";

    {
        std::ofstream f(tmp, std::ios::binary);
        f << out.c_str();
    }

    std::filesystem::rename(tmp, italk_data_path_);
}

static std::pair<std::string, YAML::Node> italk_find_fav(YAML::Node data, const std::string& fav_id) {
    auto cats = data["favorites"]["categories"];
    if (!cats || !cats.IsMap()) return {"", YAML::Node()};

    for (auto it = cats.begin(); it != cats.end(); ++it) {
        std::string cat = it->first.as<std::string>();
        YAML::Node items = it->second;
        if (!items || !items.IsSequence()) continue;
        for (size_t i = 0; i < items.size(); ++i) {
            YAML::Node fav = items[i];
            if (fav["id"] && fav["id"].as<std::string>() == fav_id) {
                return {cat, fav};
            }
        }
    }
    return {"", YAML::Node()};
}

static bool erase_favorite_by_id(YAML::Node cats, const std::string& cat, const std::string& fav_id) {
    YAML::Node items = cats[cat];
    if (!items || !items.IsSequence()) return false;

    YAML::Node new_items(YAML::NodeType::Sequence);
    bool removed = false;
    for (size_t i = 0; i < items.size(); ++i) {
        YAML::Node item = items[i];
        if (item["id"] && item["id"].as<std::string>() == fav_id) {
            removed = true;
            continue;
        }
        // Deep copy the item into the new sequence
        new_items.push_back(YAML::Clone(item));
    }
    if (removed)
        // if category is now empty create new empty node
        // otherwise we end up with [] in yaml file
        cats[cat] = new_items.size() > 0 ? new_items : YAML::Node();
    return removed;
}

static json yaml_to_json_value(const YAML::Node& n) {
    if (!n || n.IsNull()) {
        return nullptr;
    }

    if (n.IsSequence()) {
        json arr = json::array();
        for (size_t i = 0; i < n.size(); ++i) {
            arr.push_back(yaml_to_json_value(n[i]));
        }
        return arr;
    }

    if (n.IsMap()) {
        json obj = json::object();
        for (auto it = n.begin(); it != n.end(); ++it) {
            obj[it->first.as<std::string>()] = yaml_to_json_value(it->second);
        }
        return obj;
    }

    return n.as<std::string>();
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

    json j;
    if (req.method == "POST" || req.method == "PUT") {
        try {
            if (!req.body.empty()) j = json::parse(req.body);
        } catch (const std::exception& e) {
            return send_json_error(client_fd, 400, std::string("Invalid JSON: ") + e.what()), true;
        }
    }

    // GET /italk/state
    if (req.method == "GET" && subpath == "/state") {
        std::lock_guard<std::mutex> lock(italk_mutex_);
        YAML::Node data = italk_load();
        json j = yaml_to_json_value(data);
        return send_response(client_fd, 200, "application/json", j.dump()), true;
    }

    // POST /italk/settings/voice
    if (req.method == "POST" && subpath == "/settings/voice") {
        std::string voice = trim_copy(j.value("voice", ""));
        std::lock_guard<std::mutex> lock(italk_mutex_);
        YAML::Node data = italk_load();
        data["settings"]["last_voice"] = voice;
        italk_save(data);
        return send_json_ok(client_fd), true;
    }

    // POST /italk/settings/fav_category
    if (req.method == "POST" && subpath == "/settings/fav_category") {
        std::string cat = trim_copy(j.value("category", ""));
        std::lock_guard<std::mutex> lock(italk_mutex_);
        YAML::Node data = italk_load();
        data["settings"]["last_fav_category"] = cat;
        italk_save(data);
        return send_json_ok(client_fd), true;
    }

    // POST /italk/settings/expanded_categories
    if (req.method == "POST" && subpath == "/settings/expanded_categories") {
        if (!j.contains("categories") || !j["categories"].is_array()) {
            return send_json_error(client_fd, 400, "categories required"), true;
        }
        std::lock_guard<std::mutex> lock(italk_mutex_);
        YAML::Node data = italk_load();
        YAML::Node arr(YAML::NodeType::Sequence);
        for (const auto& x : j["categories"]) arr.push_back(x.get<std::string>());
        data["settings"]["expanded_fav_categories"] = arr;
        italk_save(data);
        return send_json_ok(client_fd), true;
    }

    // POST /italk/tags
    if (req.method == "POST" && subpath == "/tags") {
        std::string label = trim_copy(j.value("label", ""));
        std::string text = trim_copy(j.value("text", ""));
        if (text.empty()) return send_json_error(client_fd, 400, "text required"), true;

        std::string tag_id = "tag_" + random_hex(8);

        std::lock_guard<std::mutex> lock(italk_mutex_);
        YAML::Node data = italk_load();
        YAML::Node tag;
        tag["id"] = tag_id;
        tag["label"] = !label.empty() ? label : text.substr(0, std::min<size_t>(12, text.size()));
        tag["text"] = text;
        tag["created_at"] = italk_now_iso();
        data["tags"].push_back(tag);
        italk_save(data);
        json j;
        j["id"] = tag_id;
        return send_response(client_fd, 200, "application/json", j.dump()), true;
    }

    // PUT /italk/tags/{tag_id}
    if (req.method == "PUT" && starts_with(subpath, "/tags/")) {
        auto parts = split_path(subpath);
        if (parts.size() == 2 && parts[0] == "tags") {
            std::string tag_id = parts[1];
            std::string label = trim_copy(j.value("label", ""));
            std::string text = trim_copy(j.value("text", ""));

            std::lock_guard<std::mutex> lock(italk_mutex_);
            YAML::Node data = italk_load();
            YAML::Node tags = data["tags"];
            for (size_t i = 0; i < tags.size(); ++i) {
                YAML::Node t = tags[i];
                if (t["id"] && t["id"].as<std::string>() == tag_id) {
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
            YAML::Node data = italk_load();
            YAML::Node new_tags(YAML::NodeType::Sequence);
            YAML::Node tags = data["tags"];
            for (size_t i = 0; i < tags.size(); ++i) {
                YAML::Node t = tags[i];
                if (!(t["id"] && t["id"].as<std::string>() == tag_id)) new_tags.push_back(t);
            }
            data["tags"] = new_tags;
            italk_save(data);
            return send_json_ok(client_fd), true;
        }
    }

    // POST /italk/tags/reorder
    if (req.method == "POST" && subpath == "/tags/reorder") {
        if (!j.contains("ids") || !j["ids"].is_array()) {
            return send_json_error(client_fd, 400, "ids required"), true;
        }

        std::lock_guard<std::mutex> lock(italk_mutex_);
        YAML::Node data = italk_load();

        std::unordered_map<std::string, YAML::Node> tag_map;
        YAML::Node tags = data["tags"];
        for (size_t i = 0; i < tags.size(); ++i) {
            YAML::Node t = tags[i];
            if (t["id"]) tag_map[t["id"].as<std::string>()] = t;
        }

        YAML::Node reordered(YAML::NodeType::Sequence);
        for (const auto& idj : j["ids"]) {
            std::string tid = idj.get<std::string>();
            auto it = tag_map.find(tid);
            if (it != tag_map.end()) reordered.push_back(it->second);
        }

        data["tags"] = reordered;
        italk_save(data);
        return send_json_ok(client_fd), true;
    }

    // POST /italk/favorites
    if (req.method == "POST" && subpath == "/favorites") {
        std::string category = trim_copy(j.value("category", "Unsorted"));
        if (category.empty()) category = "Unsorted";
        std::string text = trim_copy(j.value("text", ""));
        if (text.empty()) return send_json_error(client_fd, 400, "text required"), true;

        std::string fav_id = "fav_" + random_hex(8);
        const std::string now = italk_now_iso();

        std::lock_guard<std::mutex> lock(italk_mutex_);
        YAML::Node data = italk_load();
        if (!data["favorites"]["categories"][category] || !data["favorites"]["categories"][category].IsSequence()) {
            data["favorites"]["categories"][category] = YAML::Node(YAML::NodeType::Sequence);
        }

        YAML::Node fav;
        fav["id"] = fav_id;
        fav["text"] = text;
        fav["created_at"] = now;
        fav["updated_at"] = now;
        data["favorites"]["categories"][category].push_back(fav);
        italk_save(data);
        json j;
        j["id"] = fav_id;
        return send_response(client_fd, 200, "application/json", j.dump()), true;
    }

    // PUT /italk/favorites/{fav_id}
    if (req.method == "PUT" && starts_with(subpath, "/favorites/")) {
        auto parts = split_path(subpath);
        if (parts.size() == 2 && parts[0] == "favorites") {
            std::string fav_id = parts[1];
            std::string category = trim_copy(j.value("category", "Unsorted"));
            if (category.empty()) category = "Unsorted";
            std::string text = trim_copy(j.value("text", ""));

            std::lock_guard<std::mutex> lock(italk_mutex_);
            YAML::Node data = italk_load();

            auto found = italk_find_fav(data, fav_id);
            std::string old_cat = found.first;
            YAML::Node fav = found.second;

            if (!fav || old_cat.empty()) {
                return send_json_error(client_fd, 404, "favorite not found"), true;
            }

            // Copy into a standalone node
            YAML::Node updated;
            updated["id"] = fav["id"].as<std::string>();
            updated["text"] = !text.empty() ? text : (fav["text"] ? fav["text"].as<std::string>() : "");
            updated["created_at"] = fav["created_at"] ? fav["created_at"].as<std::string>() : italk_now_iso();
            updated["updated_at"] = italk_now_iso();

            erase_favorite_by_id(data["favorites"]["categories"], old_cat, fav_id);

            if (!data["favorites"]["categories"][category] || !data["favorites"]["categories"][category].IsSequence()) {
                data["favorites"]["categories"][category] = YAML::Node(YAML::NodeType::Sequence);
            }
            data["favorites"]["categories"][category].push_back(updated);

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
            YAML::Node data = italk_load();
            auto found = italk_find_fav(data, fav_id);
            if (found.second && !found.first.empty()) {
                erase_favorite_by_id(data["favorites"]["categories"], found.first, fav_id);
                italk_save(data);
            }
            return send_json_ok(client_fd), true;
        }
    }

    // POST /italk/favorites/reorder
    if (req.method == "POST" && subpath == "/favorites/reorder") {
        std::string from_cat = j.value("from_cat", "");
        std::string to_cat = j.value("to_cat", "");
        int from_index = j.value("from", -1);
        int to_index = j.value("to", -1);

        std::lock_guard<std::mutex> lock(italk_mutex_);
        YAML::Node data = italk_load();
        YAML::Node cats = data["favorites"]["categories"];

        if (!cats[from_cat] || !cats[to_cat]) {
            return send_json_error(client_fd, 400, "Category not found"), true;
        }

        YAML::Node items_from = cats[from_cat];
        YAML::Node items_to = cats[to_cat];

        if (from_index < 0 || static_cast<size_t>(from_index) >= items_from.size()) {
            return send_json_error(client_fd, 400, "From index out of range"), true;
        }

        YAML::Node item = items_from[from_index];

        YAML::Node new_from(YAML::NodeType::Sequence);
        for (size_t i = 0; i < items_from.size(); ++i) {
            if (static_cast<int>(i) != from_index) new_from.push_back(items_from[i]);
        }

        std::vector<YAML::Node> to_vec;
        if (from_cat == to_cat) {
            for (size_t i = 0; i < new_from.size(); ++i) to_vec.push_back(new_from[i]);
        } else {
            for (size_t i = 0; i < items_to.size(); ++i) to_vec.push_back(items_to[i]);
        }

        int idx_to = std::max(0, std::min<int>(to_index, static_cast<int>(to_vec.size())));
        to_vec.insert(to_vec.begin() + idx_to, item);

        YAML::Node new_to(YAML::NodeType::Sequence);
        for (auto& v : to_vec) new_to.push_back(v);

        cats[from_cat] = new_from.size() > 0 ? new_from : YAML::Node();
        cats[to_cat] = new_to.size() > 0 ? new_to : YAML::Node();

        italk_save(data);
        return send_json_ok(client_fd), true;
    }

    // POST /italk/session/append_line
    if (req.method == "POST" && subpath == "/session/append_line") {
        std::string text = trim_copy(j.value("text", ""));
        if (text.empty()) {
            return send_json_ok(client_fd), true;
        }

        std::string line_id = "line_" + random_hex(10);
        std::string norm = normalize_spaces_lower(text);

        std::lock_guard<std::mutex> lock(italk_mutex_);
        YAML::Node data = italk_load();
        YAML::Node lines = data["last_session"]["lines"];

        YAML::Node new_lines(YAML::NodeType::Sequence);
        new_lines.push_back(make_session_line(line_id, text, italk_now_iso()));

        for (size_t i = 0; i < lines.size(); ++i) {
            YAML::Node line = lines[i];
            std::string old_text = line["text"] ? line["text"].as<std::string>() : "";
            if (normalize_spaces_lower(old_text) != norm) {
                new_lines.push_back(line);
            }
        }

        data["last_session"]["lines"] = new_lines;
        italk_save(data);
        json j;
        j["id"] = line_id;
        return send_response(client_fd, 200, "application/json", j.dump()), true;
    }

    // POST /italk/session/delete_line
    if (req.method == "POST" && subpath == "/session/delete_line") {
        std::string line_id = trim_copy(j.value("id", ""));
        if (line_id.empty()) return send_json_ok(client_fd), true;

        std::lock_guard<std::mutex> lock(italk_mutex_);
        YAML::Node data = italk_load();
        YAML::Node lines = data["last_session"]["lines"];
        YAML::Node new_lines(YAML::NodeType::Sequence);
        bool changed = false;

        for (size_t i = 0; i < lines.size(); ++i) {
            YAML::Node line = lines[i];
            if (line["id"] && line["id"].as<std::string>() == line_id) {
                changed = true;
                continue;
            }
            new_lines.push_back(line);
        }

        if (changed) {
            data["last_session"]["lines"] = new_lines;
            italk_save(data);
        }
        return send_json_ok(client_fd), true;
    }

    // POST /italk/history/delete_line
    if (req.method == "POST" && subpath == "/history/delete_line") {
        std::string session_id = trim_copy(j.value("session_id", ""));
        std::string line_id = trim_copy(j.value("line_id", ""));
        if (session_id.empty() || line_id.empty()) return send_json_ok(client_fd), true;

        std::lock_guard<std::mutex> lock(italk_mutex_);
        YAML::Node data = italk_load();
        bool changed = false;

        YAML::Node history = data["history"];
        for (size_t i = 0; i < history.size(); ++i) {
            YAML::Node sess = history[i];
            if (sess["id"] && sess["id"].as<std::string>() == session_id) {
                YAML::Node lines = sess["lines"];
                YAML::Node new_lines(YAML::NodeType::Sequence);
                size_t before = lines.size();

                for (size_t k = 0; k < lines.size(); ++k) {
                    YAML::Node line = lines[k];
                    if (!(line["id"] && line["id"].as<std::string>() == line_id)) {
                        new_lines.push_back(line);
                    }
                }

                if (new_lines.size() != before) {
                    sess["lines"] = new_lines;
                    changed = true;
                }
                break;
            }
        }

        if (changed) italk_save(data);
        return send_json_ok(client_fd), true;
    }

    // POST /italk/session/name
    if (req.method == "POST" && subpath == "/session/name") {
        std::string name = trim_copy(j.value("name", ""));
        std::lock_guard<std::mutex> lock(italk_mutex_);
        YAML::Node data = italk_load();
        data["last_session"]["name"] = name;
        italk_save(data);
        return send_json_ok(client_fd), true;
    }

    // POST /italk/session/new
    if (req.method == "POST" && subpath == "/session/new") {
        std::lock_guard<std::mutex> lock(italk_mutex_);
        YAML::Node data = italk_load();
        data["last_session"]["name"] = "";
        data["last_session"]["started_at"] = italk_now_iso();
        data["last_session"]["lines"] = YAML::Node(YAML::NodeType::Sequence);
        italk_save(data);
        return send_json_ok(client_fd), true;
    }

    // POST /italk/session/clear
    if (req.method == "POST" && subpath == "/session/clear") {
        std::lock_guard<std::mutex> lock(italk_mutex_);
        YAML::Node data = italk_load();
        data["last_session"]["name"] = "";
        data["last_session"]["started_at"] = italk_now_iso();
        data["last_session"]["lines"] = YAML::Node(YAML::NodeType::Sequence);
        italk_save(data);
        return send_json_ok(client_fd), true;
    }

    // POST /italk/session/save
    if (req.method == "POST" && subpath == "/session/save") {
        std::string name = trim_copy(j.value("name", ""));
        std::string sess_id;

        std::lock_guard<std::mutex> lock(italk_mutex_);
        YAML::Node data = italk_load();
        YAML::Node sess = data["last_session"];

        if (!name.empty()) sess["name"] = name;

        YAML::Node lines = sess["lines"];
        if (lines && lines.IsSequence() && lines.size() > 0) {
            sess_id = "sess_" + random_hex(8);

            YAML::Node hist_item;
            hist_item["id"] = sess_id;
            hist_item["name"] = sess["name"] ? sess["name"].as<std::string>() : "";
            hist_item["saved_at"] = italk_now_iso();

            YAML::Node copied_lines(YAML::NodeType::Sequence);
            for (size_t i = 0; i < lines.size(); ++i) copied_lines.push_back(lines[i]);
            hist_item["lines"] = copied_lines;

            YAML::Node new_history(YAML::NodeType::Sequence);
            new_history.push_back(hist_item);
            for (size_t i = 0; i < data["history"].size(); ++i) new_history.push_back(data["history"][i]);
            data["history"] = new_history;

            italk_save(data);
        }
        json j;
        j["id"] = sess_id;
        return send_response(client_fd, 200, "application/json", j.dump()), true;
    }

    // POST /italk/history/delete
    if (req.method == "POST" && subpath == "/history/delete") {
        std::string id = trim_copy(j.value("id", ""));
        std::lock_guard<std::mutex> lock(italk_mutex_);
        YAML::Node data = italk_load();

        YAML::Node new_history(YAML::NodeType::Sequence);
        for (size_t i = 0; i < data["history"].size(); ++i) {
            YAML::Node h = data["history"][i];
            if (!(h["id"] && h["id"].as<std::string>() == id)) new_history.push_back(h);
        }
        data["history"] = new_history;
        italk_save(data);
        return send_json_ok(client_fd), true;
    }
    return false;
}
