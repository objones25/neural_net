#include "activation_functions.hpp"
#include <stdexcept>

Eigen::MatrixXd softmax(const Eigen::MatrixXd& x) {
    Eigen::MatrixXd result(x.rows(), x.cols());
    for (int i = 0; i < x.rows(); ++i) {
        double max_val = x.row(i).maxCoeff();
        Eigen::VectorXd exp_vals = (x.row(i).array() - max_val).exp();
        double sum_exp_vals = exp_vals.sum();
        result.row(i) = exp_vals / sum_exp_vals;
    }
    return result;
}

Eigen::MatrixXd activate(const Eigen::MatrixXd& x, ActivationType activation_type) {
    switch (activation_type) {
        case ActivationType::Linear:
            return x;
        case ActivationType::ReLU:
            return x.array().max(0.0);
        case ActivationType::Sigmoid:
            return 1.0 / (1.0 + (-x.array()).exp());
        case ActivationType::Tanh:
            return x.array().tanh();
        case ActivationType::Softmax:
            return softmax(x);
        default:
            throw std::runtime_error("Unknown activation function");
    }
}

Eigen::MatrixXd activate_derivative(const Eigen::MatrixXd& x, ActivationType activation_type) {
    switch (activation_type) {
        case ActivationType::Linear:
            return Eigen::MatrixXd::Ones(x.rows(), x.cols());
        case ActivationType::ReLU:
            return (x.array() > 0.0).cast<double>();
        case ActivationType::Sigmoid: {
            Eigen::MatrixXd sigmoid = 1.0 / (1.0 + (-x.array()).exp());
            return sigmoid.array() * (1.0 - sigmoid.array());
        }
        case ActivationType::Tanh:
            return 1.0 - x.array().tanh().square();
        case ActivationType::Softmax:
            // Note: This is a simplification. The actual Jacobian of softmax is more complex.
            return Eigen::MatrixXd::Ones(x.rows(), x.cols());
        default:
            throw std::runtime_error("Unknown activation function");
    }
}