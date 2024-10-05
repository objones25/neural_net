#pragma once
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <exception>

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
private:
    static LogLevel current_level;
    static bool debug_mode;

    static std::string get_current_time() {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
        return ss.str();
    }

public:
    static void set_log_level(LogLevel level);
    static void set_debug_mode(bool mode);
    static bool get_debug_mode();
    static void log(const std::string& message, LogLevel level = LogLevel::INFO);
    static void error(const std::string& message);
    static void log_exception(const std::exception& e, const std::string& context = "");
};