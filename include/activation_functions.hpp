#pragma once
#include <Eigen/Dense>

enum class ActivationType
{
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax
};

Eigen::MatrixXd softmax(const Eigen::MatrixXd& x);
Eigen::MatrixXd activate(const Eigen::MatrixXd& x, ActivationType activation_type);
Eigen::MatrixXd activate_derivative(const Eigen::MatrixXd& x, ActivationType activation_type);