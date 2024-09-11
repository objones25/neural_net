#include "activation_functions.hpp"
#include <stdexcept>
#include <cmath>

Eigen::VectorXd activate(const Eigen::VectorXd& x, ActivationFunction func) {
    switch(func) {
        case ActivationFunction::Sigmoid:
            return 1.0 / (1.0 + (-x).array().exp());
        case ActivationFunction::ReLU:
            return x.array().max(0.0);
        case ActivationFunction::TanH:
            return x.array().tanh();
        case ActivationFunction::Softmax: {
            Eigen::VectorXd exp_x = x.array().exp();
            return exp_x.array() / exp_x.sum();
        }
        default:
            throw std::runtime_error("Invalid activation function");
    }
}

Eigen::VectorXd activate_derivative(const Eigen::VectorXd& x, ActivationFunction func) {
    switch(func) {
        case ActivationFunction::Sigmoid: {
            Eigen::VectorXd sig = activate(x, ActivationFunction::Sigmoid);
            return sig.array() * (1.0 - sig.array());
        }
        case ActivationFunction::ReLU:
            return (x.array() > 0.0).cast<double>();
        case ActivationFunction::TanH: {
            Eigen::VectorXd tanh_x = x.array().tanh();
            return 1.0 - tanh_x.array().square();
        }
        case ActivationFunction::Softmax: {
            // Note: This is the element-wise derivative, not the full Jacobian
            Eigen::VectorXd softmax = activate(x, ActivationFunction::Softmax);
            return softmax.array() * (1.0 - softmax.array());
        }
        default:
            throw std::runtime_error("Invalid activation function");
    }
}