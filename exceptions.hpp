#pragma once

#include <stdexcept>
#include <string>

class NetworkConfigurationError : public std::runtime_error {
public:
    NetworkConfigurationError(const std::string& what_arg) : std::runtime_error(what_arg) {}
};

class TrainingDataError : public std::runtime_error {
public:
    TrainingDataError(const std::string& what_arg) : std::runtime_error(what_arg) {}
};