#include "logger.hpp"

bool Logger::debug_mode = false;

void Logger::set_debug_mode(bool mode) {
    debug_mode = mode;
}

bool Logger::get_debug_mode() {
    return debug_mode;
}

void Logger::log(const std::string& message) {
    if (debug_mode) {
        std::cout << "[" << get_current_time() << "] " << message << std::endl;
    }
}

void Logger::error(const std::string& message) {
    std::cerr << "[" << get_current_time() << "] ERROR: " << message << std::endl;
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