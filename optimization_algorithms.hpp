#pragma once

#include <Eigen/Dense>
#include "activation_functions.hpp"

enum class OptimizationAlgorithm {
    GradientDescent = 0,
    Adam = 1,
    RMSprop = 2
};

void update_weights_and_biases(OptimizationAlgorithm algo,
                               Eigen::MatrixXd &w, Eigen::VectorXd &b,
                               const Eigen::MatrixXd &dw, const Eigen::VectorXd &db,
                               Eigen::MatrixXd &m_w, Eigen::VectorXd &m_b,
                               Eigen::MatrixXd &v_w, Eigen::VectorXd &v_b,
                               double learning_rate, double beta1, double beta2, double epsilon,
                               int &t);