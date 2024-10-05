#include "logger.hpp"

LogLevel Logger::current_level = LogLevel::INFO;
bool Logger::debug_mode = false;

void Logger::set_log_level(LogLevel level) {
    current_level = level;
}

void Logger::set_debug_mode(bool mode) {
    debug_mode = mode;
    current_level = mode ? LogLevel::DEBUG : LogLevel::INFO;
}

bool Logger::get_debug_mode() {
    return debug_mode;
}

void Logger::log(const std::string& message, LogLevel level) {
    if (level >= current_level) {
        std::cout << "[" << get_current_time() << "] ";
        switch (level) {
            case LogLevel::DEBUG: std::cout << "DEBUG: "; break;
            case LogLevel::INFO: std::cout << "INFO: "; break;
            case LogLevel::WARNING: std::cout << "WARNING: "; break;
            case LogLevel::ERROR: std::cout << "ERROR: "; break;
        }
        std::cout << message << std::endl;
    }
}

void Logger::error(const std::string& message) {
    log(message, LogLevel::ERROR);
}

void Logger::log_exception(const std::exception& e, const std::string& context) {
    std::stringstream ss;
    ss << "Exception caught";
    if (!context.empty()) {
        ss << " in " << context;
    }
    ss << ": " << e.what();
    error(ss.str());
}