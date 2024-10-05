#pragma once

#include <stdexcept>
#include <string>

class NeuralNetworkException : public std::runtime_error
{
public:
    NeuralNetworkException(const std::string &what_arg) : std::runtime_error(what_arg) {}
};

class NetworkConfigurationError : public NeuralNetworkException
{
public:
    NetworkConfigurationError(const std::string &what_arg) : NeuralNetworkException(what_arg) {}
};

class TrainingDataError : public NeuralNetworkException
{
public:
    TrainingDataError(const std::string &what_arg) : NeuralNetworkException(what_arg) {}
};

class NumericalInstabilityError : public NeuralNetworkException
{
public:
    NumericalInstabilityError(const std::string &what_arg) : NeuralNetworkException(what_arg) {}
};

class GradientExplodingError : public NeuralNetworkException
{
public:
    GradientExplodingError(const std::string &what_arg) : NeuralNetworkException(what_arg) {}
};

class WeightInitializationError : public NeuralNetworkException
{
public:
    WeightInitializationError(const std::string &what_arg) : NeuralNetworkException(what_arg) {}
};

class SizeMismatchError : public NeuralNetworkException
{
public:
    SizeMismatchError(const std::string &what_arg) : NeuralNetworkException(what_arg) {}
};

class BatchNormalizationError : public NeuralNetworkException
{
public:
    BatchNormalizationError(const std::string &what_arg) : NeuralNetworkException(what_arg) {}
};

class OptimizerError : public NeuralNetworkException
{
public:
    OptimizerError(const std::string &what_arg) : NeuralNetworkException(what_arg) {}
};