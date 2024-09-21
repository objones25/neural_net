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

class NumericalInstabilityError : public std::runtime_error {
public:
    NumericalInstabilityError(const std::string& what_arg) : std::runtime_error(what_arg) {}
};

class GradientExplodingError : public std::runtime_error {
public:
    GradientExplodingError(const std::string& what_arg) : std::runtime_error(what_arg) {}
};

class WeightInitializationError : public std::runtime_error {
public:
    WeightInitializationError(const std::string& what_arg) : std::runtime_error(what_arg) {}
};

class SizeMismatchError : public std::runtime_error {
public:
    SizeMismatchError(const std::string& what_arg) : std::runtime_error(what_arg) {}
};