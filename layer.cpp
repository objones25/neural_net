#include "layer.hpp"

Layer::Layer(int input_size, int output_size, bool use_batch_norm)
    : weights(Eigen::MatrixXd::Random(output_size, input_size)),
      biases(Eigen::VectorXd::Random(output_size)),
      bn_gamma_grad(Eigen::VectorXd::Zero(output_size)),
      bn_beta_grad(Eigen::VectorXd::Zero(output_size))
{
    if (use_batch_norm) {
        batch_norm = std::make_unique<BatchNorm>(output_size);
    }
}