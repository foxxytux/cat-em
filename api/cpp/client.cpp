// CodeAgent-RWKV C++ client
// Build: g++ -std=c++17 -o codeagent client.cpp -lcurl
// Usage: ./codeagent "hello world in python"

#include <iostream>
#include <string>
#include <cstdlib>
#include <curl/curl.h>

const std::string API_URL = "http://127.0.0.1:8080/v1/chat";

static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

std::string chat(const std::string& prompt) {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl init failed");

    std::string body = R"({
        "prompt": ")" + prompt + R"(",
        "system": "You are a helpful coding assistant. Think step by step.",
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9
    })";

    std::string response;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, API_URL.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        throw std::runtime_error("curl error: " + std::string(curl_easy_strerror(res)));
    }
    return response;
}

int main(int argc, char** argv) {
    std::string prompt = "Write a hello world in Python";
    if (argc > 1) {
        prompt = argv[1];
        for (int i = 2; i < argc; i++) {
            prompt += " " + std::string(argv[i]);
        }
    }

    std::cout << "CodeAgent > " << prompt << "\n\n";
    try {
        std::string resp = chat(prompt);
        // Simple JSON extraction (avoid full parser dep)
        auto text_start = resp.find("\"text\":\"");
        if (text_start != std::string::npos) {
            text_start += 8;
            auto text_end = resp.find("\",\"usage\"", text_start);
            if (text_end != std::string::npos) {
                std::string text = resp.substr(text_start, text_end - text_start);
                // Unescape basic escapes
                for (size_t i = 0; i < text.length(); i++) {
                    if (text[i] == '\\' && i+1 < text.length()) {
                        if (text[i+1] == 'n') { text.replace(i, 2, "\n"); i--; }
                        else if (text[i+1] == 't') { text.replace(i, 2, "\t"); i--; }
                        else if (text[i+1] == '"') { text.replace(i, 2, "\""); i--; }
                    }
                }
                std::cout << text << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
