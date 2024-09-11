#pragma once

#include <Eigen/Dense>

enum class ActivationFunction { Sigmoid, ReLU, TanH, Softmax };

Eigen::VectorXd activate(const Eigen::VectorXd& x, ActivationFunction func);
Eigen::VectorXd activate_derivative(const Eigen::VectorXd& x, ActivationFunction func);