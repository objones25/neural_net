#pragma once

#include <Eigen/Dense>
#include <memory>
#include "batch_normalization.hpp"

class Layer {
public:
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    std::unique_ptr<BatchNorm> batch_norm;
    Eigen::VectorXd bn_gamma_grad;
    Eigen::VectorXd bn_beta_grad;

    Layer(int input_size, int output_size, bool use_batch_norm);
    
    // You might want to add more methods here as needed
};