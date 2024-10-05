#include "activation_functions.hpp"
#include <stdexcept>

Eigen::MatrixXd softmax(const Eigen::MatrixXd &x)
{
    Eigen::MatrixXd exp_x = x.array().exp();
    return (exp_x.array().colwise() / exp_x.rowwise().sum().array()).matrix();
}

Eigen::MatrixXd activate(const Eigen::MatrixXd &x, ActivationType activation_type)
{
    Logger::log("Activate function called", LogLevel::DEBUG);
    Eigen::MatrixXd result = x; // Initialize result with the same shape as input
    switch (activation_type)
    {
    case ActivationType::Linear:
        // result is already equal to x
        break;
    case ActivationType::ReLU:
        result = result.array().max(0.0);
        break;
    case ActivationType::Sigmoid:
        result = 1.0 / (1.0 + (-result.array()).exp());
        break;
    case ActivationType::Tanh:
        result = result.array().tanh();
        break;
    case ActivationType::Softmax:
        result = softmax(result);
        break;
    default:
        Logger::error("Unknown activation function");
        throw std::runtime_error("Unknown activation function");
    }
    return result;
}

Eigen::MatrixXd activate_derivative(const Eigen::MatrixXd &x, ActivationType activation_type)
{
    Logger::log("Activate derivative function called", LogLevel::DEBUG);
    switch (activation_type)
    {
    case ActivationType::Linear:
        return Eigen::MatrixXd::Ones(x.rows(), x.cols());
    case ActivationType::ReLU:
        return (x.array() > 0.0).cast<double>();
    case ActivationType::Sigmoid:
    {
        Eigen::MatrixXd sigmoid = 1.0 / (1.0 + (-x.array()).exp());
        return sigmoid.array() * (1.0 - sigmoid.array());
    }
    case ActivationType::Tanh:
        return 1.0 - x.array().tanh().square();
    case ActivationType::Softmax:
        // Note: This is a simplification. The actual Jacobian of softmax is more complex.
        return Eigen::MatrixXd::Ones(x.rows(), x.cols());
    default:
        Logger::error("Unknown activation function in derivative");
        throw std::runtime_error("Unknown activation function in derivative");
    }
}