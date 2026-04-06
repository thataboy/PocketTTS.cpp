// PocketTTS executable
// Refactored from the original single-file implementation.

// ── Platform (must come first — winsock2.h before windows.h) ────────────────

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #ifndef _CRT_SECURE_NO_WARNINGS
    #define _CRT_SECURE_NO_WARNINGS
  #endif
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #include <direct.h>
  #include <io.h>
  #include <fcntl.h>
  #pragma comment(lib, "ws2_32.lib")
  #define ptt_mkdir(path) _mkdir(path)
  #define ptt_close closesocket
  typedef SOCKET ptt_socket_t;
  typedef int socklen_t;
  static constexpr ptt_socket_t PTT_INVALID_SOCKET = INVALID_SOCKET;
  using ssize_t = ptrdiff_t;
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  #define ptt_mkdir(path) mkdir(path, 0755)
  #define ptt_close close
  typedef int ptt_socket_t;
  static constexpr ptt_socket_t PTT_INVALID_SOCKET = -1;
  #include <signal.h>
#endif

#include "ptts_engine.cpp"

#include <atomic>
#include <filesystem>
#include <random>
#include <sstream>
#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;

namespace pocket_tts {
// ════════════════════════════════════════════════════════════════════════════
// HTTP Server
// ════════════════════════════════════════════════════════════════════════════

static std::atomic<bool> g_server_running{true};
static ptt_socket_t g_server_fd = PTT_INVALID_SOCKET;

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;

    static HttpRequest parse(ptt_socket_t client_fd) {
        HttpRequest req;
        std::string data;
        char buf[4096];

        while (true) {
            ssize_t n = recv(client_fd, buf, (int)sizeof(buf), 0);
            if (n <= 0) break;
            data.append(buf, n);

            size_t header_end = data.find("\r\n\r\n");
            if (header_end != std::string::npos) {
                // Case-insensitive search for Content-Length header
                std::string lower_data = data.substr(0, header_end);
                for (auto& c : lower_data) c = std::tolower((unsigned char)c);
                size_t cl_pos = lower_data.find("content-length:");

                if (cl_pos != std::string::npos) {
                    size_t cl_end = data.find("\r\n", cl_pos);
                    int content_length = std::stoi(data.substr(cl_pos + 15, cl_end - cl_pos - 15));
                    size_t body_start = header_end + 4;

                    while (data.size() < body_start + content_length) {
                        n = recv(client_fd, buf, (int)sizeof(buf), 0);
                        if (n <= 0) break;
                        data.append(buf, n);
                    }
                }
                break;
            }
        }

        size_t line_end = data.find("\r\n");
        if (line_end != std::string::npos) {
            std::string line = data.substr(0, line_end);
            size_t sp1 = line.find(' ');
            size_t sp2 = line.find(' ', sp1 + 1);
            if (sp1 != std::string::npos && sp2 != std::string::npos) {
                req.method = line.substr(0, sp1);
                req.path = line.substr(sp1 + 1, sp2 - sp1 - 1);
            }
        }

        size_t body_start = data.find("\r\n\r\n");
        if (body_start != std::string::npos) {
            req.body = data.substr(body_start + 4);
        }

        return req;
    }
};

// Helper to create a standard PCM16 (Format 1) mono WAV buffer
static std::vector<uint8_t> wav_encode(const float* samples, size_t n, int sample_rate) {
    size_t data_size = n * 2;
    size_t file_size = 36 + data_size;
    std::vector<uint8_t> buf(44 + data_size);
    auto w = [&](size_t off, const char* s, size_t len) { memcpy(buf.data() + off, s, len); };
    auto w32 = [&](size_t off, uint32_t v) { memcpy(buf.data() + off, &v, 4); };
    auto w16 = [&](size_t off, uint16_t v) { memcpy(buf.data() + off, &v, 2); };

    w(0, "RIFF", 4); w32(4, file_size); w(8, "WAVE", 4);
    w(12, "fmt ", 4); w32(16, 16);
    w16(20, 1); // PCM Format 1
    w16(22, 1); // Mono
    w32(24, sample_rate);
    w32(28, sample_rate * 2); // Byte rate
    w16(32, 2); // Block align
    w16(34, 16); // 16 bits
    w(36, "data", 4); w32(40, data_size);

    int16_t* pcm = reinterpret_cast<int16_t*>(buf.data() + 44);
    for (size_t i = 0; i < n; ++i) {
        float s = std::max(-1.0f, std::min(1.0f, samples[i]));
        pcm[i] = static_cast<int16_t>(s * 32767.0f);
    }
    return buf;
}

static std::string get_mime_type(const std::string& path) {
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return "application/octet-stream";
    std::string ext = path.substr(dot);
    for (auto& c : ext) c = tolower(c);
    if (ext == ".html" || ext == ".htm") return "text/html";
    if (ext == ".xhtml") return "text/xhtml";
    if (ext == ".css") return "text/css";
    if (ext == ".txt" || ext == ".py" || ext == ".rs" || ext == ".js") return "text/plain";
    if (ext == ".pdf") return "application/pdf";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".png") return "image/png";
    if (ext == ".svg") return "image/svg";
    if (ext == ".gif") return "image/gif";
    if (ext == ".webp") return "image/webp";
    if (ext == ".wav") return "audio/wav";
    if (ext == ".mp3") return "audio/mpeg";
    if (ext == ".ogg") return "audio/ogg";
    if (ext == ".csv") return "text/csv";
    if (ext == ".mkv") return "video/x-matroska";
    if (ext == ".mp4") return "video/mp4";
    if (ext == ".mov") return "video/quicktime";
    return "application/octet-stream";
}

static std::string url_decode(const std::string& str) {
    std::string res;
    for (size_t i = 0; i < str.length(); ++i) {
        if (str[i] == '%' && i + 2 < str.length()) {
            int value;
            sscanf(str.substr(i + 1, 2).c_str(), "%x", &value);
            res += static_cast<char>(value);
            i += 2;
        } else if (str[i] == '+') res += ' ';
        else res += str[i];
    }
    return res;
}

class TTSServer {
    PocketTTS& tts_;
    int port_;
    ptt_socket_t server_fd_ = PTT_INVALID_SOCKET;
    std::mutex tts_mutex_;
    std::vector<std::string> voice_names_;

    const std::vector<std::pair<std::string, std::string>> route_map = {
        {"/books", "/Volumes/T7/books"},
        {"/italk", "./italk"},
        {"/", "/Volumes/T7/downloads"},
    };

    void scan_voices() {
        voice_names_.clear();
        DIR* dir = opendir(tts_.config().voices_dir.c_str());
        if (!dir) return;
        struct dirent* ent;
        std::string voice;
        std::cout << "Loading voices..." << std::flush;
        while ((ent = readdir(dir)) != nullptr) {
            std::string n = ent->d_name;
            if (n.size() > 4 && n.substr(n.size() - 4) == ".wav") {
                voice = n.substr(0, n.size() - 4);
                std::cout << " " << voice << std::flush;
                // tts_.prepare_voice(voice);
                voice_names_.push_back(voice);
            }
        }
        closedir(dir);
        std::cout << std::endl;
        std::sort(voice_names_.begin(), voice_names_.end());
    }

public:
    TTSServer(PocketTTS& tts, int port) : tts_(tts), port_(port) {
        scan_voices();
    }

    ~TTSServer() {
        if (server_fd_ != PTT_INVALID_SOCKET && server_fd_ == g_server_fd) {
            ptt_close(server_fd_);
            g_server_fd = PTT_INVALID_SOCKET;
        }
        server_fd_ = PTT_INVALID_SOCKET;
#ifdef _WIN32
        WSACleanup();
#endif
    }

    bool start() {
#ifdef _WIN32
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
            std::cerr << "WSAStartup failed\n";
            return false;
        }
#endif
        server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd_ == PTT_INVALID_SOCKET) {
            std::cerr << "Failed to create socket\n";
            return false;
        }
        g_server_fd = server_fd_;

        int opt = 1;
        setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port_);

        if (bind(server_fd_, (sockaddr*)&addr, sizeof(addr)) < 0) {
            std::cerr << "Failed to bind to port " << port_ << "\n";
            return false;
        }

        if (listen(server_fd_, 5) < 0) {
            std::cerr << "Failed to listen\n";
            return false;
        }

        std::cout << "TTS Server listening on http://localhost:" << port_ << "\n";
        std::cout << "Endpoints:\n";
        std::cout << "  POST /stream                         - Streaming TTS\n";
        std::cout << "  POST /synthesize                     - Complete TTS audio\n";
        std::cout << "  GET  /voices                         - Get list of voices\n";
        std::cout << "  GET  /voices/refresh                 - Refresh voices\n";
        std::cout << "  GET  /health                         - Health check\n";
        std::cout << "  GET  /italk                          - iTalk app\n";
        std::cout << "Press Ctrl+C to stop\n\n";

        return true;
    }

    void run() {
        while (g_server_running) {
            sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);

            ptt_socket_t client_fd = accept(server_fd_, (sockaddr*)&client_addr, &client_len);
            if (client_fd == PTT_INVALID_SOCKET) break;

#ifdef _WIN32
            DWORD tv = 30000;
            setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
#else
            struct timeval tv;
            tv.tv_sec = 30;
            tv.tv_usec = 0;
            setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif

            handle_request(client_fd);
            ptt_close(client_fd);
        }
    }

private:
    static bool ptt_send(ptt_socket_t fd, const void* data, size_t len) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags |= MSG_NOSIGNAL;
#endif
        const char* ptr = static_cast<const char*>(data);
        while (len > 0) {
            ssize_t sent = send(fd, ptr, static_cast<int>(len), flags);
            if (sent <= 0) return false;
            ptr += sent;
            len -= static_cast<size_t>(sent);
        }
        return true;
    }

    bool send_response(ptt_socket_t fd, int status, const std::string& content_type, const std::string& body) {
        std::string status_text =
            (status == 200) ? "OK" :
            (status == 301) ? "Moved Permanently" :
            (status == 400) ? "Bad Request" :
            (status == 404) ? "Not Found" :
            "Error";

        std::ostringstream resp;
        resp << "HTTP/1.1 " << status << " " << status_text << "\r\n";
        resp << "Content-Type: " << content_type << "\r\n";
        resp << "Content-Length: " << body.size() << "\r\n";
        resp << "Access-Control-Allow-Origin: *\r\n";
        resp << "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n";
        resp << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";
        resp << "\r\n";
        resp << body;

        std::string data = resp.str();
        return ptt_send(fd, data.c_str(), data.size());
    }

    bool send_binary_response(ptt_socket_t fd, const std::string& content_type, const std::vector<uint8_t>& body) {
        return send_binary_response(fd, content_type, body.data(), body.size());
    }

    bool send_binary_response(ptt_socket_t fd, const std::string& content_type, const void* data, size_t len) {
        std::ostringstream resp;
        resp << "HTTP/1.1 200 OK\r\n";
        resp << "Content-Type: " << content_type << "\r\n";
        resp << "Content-Length: " << len << "\r\n";
        resp << "Access-Control-Allow-Origin: *\r\n";
        resp << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";
        resp << "\r\n";

        std::string header = resp.str();
        return ptt_send(fd, header.c_str(), header.size()) && ptt_send(fd, data, len);
    }

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

    void render_dir_list(
        ptt_socket_t client_fd,
        const std::string& decoded_path,
        const std::string& local_path,
        bool at_root
    ) {
        std::vector<std::pair<std::string, bool>> items;
        DIR* dir = opendir(local_path.c_str());
        if (dir) {
            struct dirent* ent;
            while ((ent = readdir(dir))) {
                std::string name = ent->d_name;
                if (name.empty() || name[0] == '.') continue;
                items.push_back({name, ent->d_type == DT_DIR});
            }
            closedir(dir);
        }
        std::sort(items.begin(), items.end());

        std::ostringstream html;
        html << "<html><head><meta charset='utf-8'></head><body><h1>Index of " << decoded_path << "</h1><ul>";
        if (!at_root) {
            html << "<li style=\"margin: 1em 1em\"><a href=\"..\">[.. parent folder]</a></li>";
        }
        for (const auto& item : items) {
            html << "<li style=\"margin: 1em 1em\"><a href=\"" << item.first << (item.second ? "/" : "") << "\">"
                 << item.first << (item.second ? "/" : "") << "</a></li>";
        }
        html << "</ul></body></html>";
        send_response(client_fd, 200, "text/html", html.str());
    }

    bool send_chunked_header(ptt_socket_t fd, const std::string& content_type) {
        std::ostringstream resp;
        resp << "HTTP/1.1 200 OK\r\n";
        resp << "Content-Type: " << content_type << "\r\n";
        resp << "Transfer-Encoding: chunked\r\n";
        resp << "Access-Control-Allow-Origin: *\r\n";
        resp << "\r\n";

        std::string data = resp.str();
        return ptt_send(fd, data.c_str(), data.size());
    }

    bool send_chunk(ptt_socket_t fd, const void* data, size_t len) {
        char size_buf[32];
        snprintf(size_buf, sizeof(size_buf), "%zx\r\n", len);
        return ptt_send(fd, size_buf, strlen(size_buf))
            && ptt_send(fd, data, len)
            && ptt_send(fd, "\r\n", 2);
    }

    bool send_final_chunk(ptt_socket_t fd) {
        return ptt_send(fd, "0\r\n\r\n", 5);
    }

    bool is_socket_alive(ptt_socket_t fd) {
        char buf;
        int res = recv(fd, &buf, 1, MSG_PEEK | MSG_DONTWAIT);
        if (res > 0) return true;
        if (res == 0) return false;
#ifdef _WIN32
        int err = WSAGetLastError();
        return err == WSAEWOULDBLOCK;
#else
        if (errno == EAGAIN || errno == EWOULDBLOCK) return true;
        return false;
#endif
    }

    #include "italk.hpp"

    void handle_request(int client_fd) {
        HttpRequest req = HttpRequest::parse(client_fd);
        if (req.path.empty()) return;

        if (req.method == "GET" && req.path == "/health") {
            send_json_ok(client_fd);
            return;
        }

        // /stream
        if (req.method == "POST" && req.path == "/stream") {
            try {
                auto j = json::parse(req.body);
                std::string text = j.value("text", "");
                std::string voice = j.value("voice", "narrator");

                if (text.empty() || voice.empty()) {
                    send_json_error(client_fd, 400, "Missing text or voice");
                    return;
                }

                std::cout << "\n⏩" << voice << "➡️" << text.substr(0, 150)
                          << (text.size() > 150 ? "..." : "") << "⬅️\n";

                auto start = std::chrono::high_resolution_clock::now();

                send_chunked_header(client_fd, "audio/pcm;rate=24000");

                bool first_chunk = true;
                bool client_disconnected = false;
                size_t total_samples = 0;
                double latency = 0.0;

                {
                    std::lock_guard<std::mutex> lock(tts_mutex_);
                    tts_.stream(text, voice, [&](const float* s, size_t n) {
                        std::vector<int16_t> pcm(n);
                        for (size_t i = 0; i < n; ++i) {
                            pcm[i] = static_cast<int16_t>(std::max(-1.0f, std::min(1.0f, s[i])) * 32767.0f);
                        }
                        if (!is_socket_alive(client_fd) || !send_chunk(client_fd, pcm.data(), n * 2)) {
                            client_disconnected = true;
                            return false;
                        }
                        if (first_chunk) {
                            auto now = std::chrono::high_resolution_clock::now();
                            latency = std::chrono::duration<double, std::milli>(now - start).count();
                            first_chunk = false;
                        }
                        total_samples += n;
                        return true;
                    });
                }

                if (client_disconnected) {
                    std::cout << "  Client disconnected during stream\n";
                    return;
                } else {
                    send_final_chunk(client_fd);
                }

                auto end = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(end - start).count();
                double duration = double(total_samples) / PocketTTS::SR;
                std::cout << "     len: " << std::fixed << std::setprecision(2)
                                          << duration << "s (" << text.size() << ")"
                          << " | latency: " << std::fixed << std::setprecision(0) << latency << "ms"
                          << " | time: " << std::fixed << std::setprecision(2) << elapsed << "s"
                          << " | speed: " << (elapsed > 0 ? duration / elapsed : 0) << "x\n";
            } catch (const std::exception& e) {
                send_json_error(client_fd, 400, e.what());
            }
            return;
        }

        // /synthesize
        if (req.method == "POST" && req.path == "/synthesize") {
            try {
                auto j = json::parse(req.body);
                std::string text = j.value("text", "");
                std::string voice = j.value("voice", "narrator");

                if (text.empty()) {
                    send_response(client_fd, 400, "text/plain", "Missing text");
                    return;
                }

                std::cout << voice << "➡️" << text.substr(0, 100)
                          << (text.size() > 100 ? "..." : "") << "⬅️\n";
                auto start = std::chrono::high_resolution_clock::now();

                std::vector<float> all_samples;
                bool client_disconnected = false;

                {
                    std::lock_guard<std::mutex> lock(tts_mutex_);
                    tts_.stream(text, voice, [&](const float* samples, size_t n) {
                        if (!is_socket_alive(client_fd)) {
                            client_disconnected = true;
                            return false;
                        }
                        all_samples.insert(all_samples.end(), samples, samples + n);
                        return true;
                    });
                }

                if (client_disconnected) {
                    std::cout << "  Client disconnected during synthesis.\n";
                    return;
                }

                auto wav = wav_encode(all_samples.data(), all_samples.size(), PocketTTS::SR);
                auto end = std::chrono::high_resolution_clock::now();
                if (send_binary_response(client_fd, "audio/wav", wav)) {
                    double elapsed = std::chrono::duration<double>(end - start).count();
                    double duration = static_cast<double>(all_samples.size()) / 24000.0;
                    std::cout << "     len: " << std::fixed << std::setprecision(2)
                                              << duration << "s (" << text.size() << ")"
                              << " | time: " << std::fixed << std::setprecision(2) << elapsed << "s"
                              << " | speed: " << (elapsed > 0 ? duration / elapsed : 0) << "x\n";
                }
            } catch (const std::exception& e) {
                send_json_error(client_fd, 400, e.what());
            }
            return;
        }

        // /generate
        /*
        if (req.method == "POST" && req.path == "/generate") {
            try {
                auto j = json::parse(req.body);
                std::string text = j.value("text", "");
                std::string voice = j.value("voice", "narrator");

                if (text.empty()) {
                    send_response(client_fd, 400, "text/plain", "Missing text");
                    return;
                }

                std::cout << voice << "➡️" << text << "⬅️\n";
                auto start = std::chrono::high_resolution_clock::now();

                bool client_disconnected = false;

                AudioData audio;
                {
                    std::lock_guard<std::mutex> lock(tts_mutex_);
                    audio = tts_.generate(text, voice);
                }

                if (!is_socket_alive(client_fd)) {
                    std::cout << "  Client disconnected during synthesis.\n";
                    return;
                }

                auto wav = wav_encode(audio.samples.data(), audio.samples.size(), PocketTTS::SR);
                auto end = std::chrono::high_resolution_clock::now();
                if (send_binary_response(client_fd, "audio/wav", wav)) {
                    double elapsed = std::chrono::duration<double>(end - start).count();
                    double duration = audio.duration_sec();
                    std::cout << "     len: " << std::fixed << std::setprecision(2)
                                              << duration << "s (" << text.size() << ")"
                              << " | time: " << std::fixed << std::setprecision(2) << elapsed << "s"
                              << " | speed: " << (elapsed > 0 ? duration / elapsed : 0) << "x\n";
                }
            } catch (const std::exception& e) {
                send_json_error(client_fd, 400, e.what());
            }
            return;
        }
        */

        // /voices
        if (req.method == "GET" && (req.path == "/voices" || req.path == "/voices/refresh")) {
            if (req.path == "/voices/refresh") {
                scan_voices();
                tts_.clear_vcache();
            }
            json j;
            j["voices"] = voice_names_;
            send_response(client_fd, 200, "application/json", j.dump());
            return;
        }

        if (handle_italk_request(client_fd, req)) {
            return;
        }

        // Static files
        if (req.method == "GET") {
            std::string decoded_path = url_decode(req.path);
            std::string local_path;
            std::string matched_prefix;
            bool at_root = false;

            for (const auto& [prefix, root] : route_map) {
                if (decoded_path.rfind(prefix, 0) == 0) {
                    matched_prefix = prefix;
                    std::string subpath = decoded_path.substr(prefix.length());
                    if (subpath.empty() || subpath == "/") at_root = true;

                    local_path = root;
                    if (!subpath.empty()) {
                        if (local_path.back() != '/' && subpath[0] != '/') local_path += "/";
                        local_path += subpath;
                    }
                    break;
                }
            }

            if (local_path.empty()) {
                send_response(client_fd, 404, "text/plain", "Not Found");
                return;
            }

            struct stat st;
            if (stat(local_path.c_str(), &st) != 0) {
                send_response(client_fd, 404, "text/plain", "Not Found");
                return;
            }

            if (S_ISDIR(st.st_mode)) {
                if (decoded_path.back() != '/') {
                    std::string redir = "HTTP/1.1 301 Moved Permanently\r\nLocation: " + req.path + "/\r\nContent-Length: 0\r\n\r\n";
                    send(client_fd, redir.data(), redir.size(), 0);
                    return;
                }

                for (const std::string& index_file : {"index.html", "index.htm"}) {
                    std::string index_path = local_path + (local_path.back() == '/' ? "" : "/") + index_file;
                    struct stat ist;
                    if (stat(index_path.c_str(), &ist) == 0) {
                        local_path = index_path;
                        goto serve_file_label;
                    }
                }

                render_dir_list(client_fd, decoded_path, local_path, at_root);
                return;
            }

        serve_file_label:
            std::ifstream f(local_path, std::ios::binary);
            if (!f) {
                send_response(client_fd, 404, "text/plain", "File not readable");
                return;
            }

            f.seekg(0, std::ios::end);
            std::streamsize file_size = f.tellg();
            f.seekg(0, std::ios::beg);

            std::ostringstream resp;
            resp << "HTTP/1.1 200 OK\r\n"
                 << "Content-Type: " << get_mime_type(local_path) << "\r\n"
                 << "Content-Length: " << file_size << "\r\n"
                 << "Access-Control-Allow-Origin: *\r\n\r\n";

            std::string header = resp.str();
            if (!ptt_send(client_fd, header.c_str(), header.size())) return;

            char buffer[65536];
            while (f.read(buffer, sizeof(buffer)).gcount() > 0) {
                if (!ptt_send(client_fd, buffer, static_cast<size_t>(f.gcount()))) break;
            }
            return;
        }

        send_response(client_fd, 404, "text/plain", "Not Found");
    }
};


} // namespace pocket_tts

#ifndef PTT_SHARED_LIB

static void signal_handler(int sig) {
    (void)sig;
    pocket_tts::g_server_running = false;
    if (pocket_tts::g_server_fd != PTT_INVALID_SOCKET) {
        ptt_close(pocket_tts::g_server_fd);
        pocket_tts::g_server_fd = PTT_INVALID_SOCKET;
    }
    std::cout << "\nShutting down...\n";
}

static int serve(const pocket_tts::Config& cfg, int server_port) {
    try {
        int threads = cfg.num_threads ? cfg.num_threads : std::max(2, int(std::thread::hardware_concurrency()) / 2);
        std::cerr << "Loading (precision=" << cfg.precision << ", threads=" << threads << ")...\n";

        auto t0 = std::chrono::high_resolution_clock::now();
        pocket_tts::PocketTTS tts(cfg);
        auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count(); };

        std::cerr << "  Loaded in " << std::fixed << std::setprecision(2) << elapsed() << "s\n";
        double warmup_ms = tts.warmup();
        std::cerr << "  Warmup in " << std::fixed << std::setprecision(0) << warmup_ms << "ms\n";

        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
#ifndef _WIN32
        signal(SIGPIPE, SIG_IGN);
#endif

        pocket_tts::TTSServer server(tts, server_port);
        if (!server.start()) return 1;
        server.run();

        if (pocket_tts::g_prof.enabled) tts.print_profiling_report();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

static int generate(
    const pocket_tts::Config& cfg,
    const std::string& text,
    const std::string& voice,
    const std::string& output) {

    bool stdout_output = output.empty();

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

int main(int argc, char* argv[]) {
    pocket_tts::Config cfg;
    int server_port = 9000;
    std::string text, voice, output;
    int nargs = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> char* {
            if (++i >= argc) { std::cerr << "Missing value for " << a << "\n"; std::exit(1); }
            return argv[i];
        };

        if (a == "-h" || a == "--help") {
            std::cerr
                << "\nUsage:\n"
                << "   Start Server:    " << argv[0] << " [OPTIONS]\n"
                << "   Generate TTS:    " << argv[0] << " [OPTIONS] TEXT VOICE [OUTPUT]\n\n"
                << "Options:\n"
                << "  --port <port>            Server port (default: 9000)\n"
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
                << "  --verbose                Enable verbose output\n"
                << "  --profile                Show profiling report\n";
            return 0;
        }
        else if (a == "--port") server_port = std::stoi(next());
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
        else if (a == "--verbose") cfg.verbose = true;
        else if (a == "--profile") pocket_tts::g_prof.enabled = true;
        else if (!a.empty() && a[0] == '-') { std::cerr << "Unknown: " << a << "\n"; return 1; }
        else {
            if (nargs == 0) text = a;
            else if (nargs == 1) voice = a;
            else if (nargs == 2) output = a;
            ++nargs;
        }
    }

    if (nargs == 0) {
        return serve(cfg, server_port);
    }
    if (nargs < 2) {
        std::cerr << "Need: TEXT VOICE [OUTPUT]\n";
        return 1;
    }
    if (output.empty()) {
        cfg.verbose = false;
        pocket_tts::g_prof.enabled = false;
    }
    return generate(cfg, text, voice, output);
}

#endif // PTT_SHARED_LIB
