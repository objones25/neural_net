#pragma once
#include <Eigen/Dense>
#include "logger.hpp"

enum class ActivationType
{
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    _COUNT
};

Eigen::MatrixXd softmax(const Eigen::MatrixXd& x);
Eigen::MatrixXd activate(const Eigen::MatrixXd& x, ActivationType activation_type);
Eigen::MatrixXd activate_derivative(const Eigen::MatrixXd& x, ActivationType activation_type);