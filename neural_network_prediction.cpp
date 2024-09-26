#include "neural_network.hpp"
#include "neural_network_common.hpp"

Eigen::VectorXd NeuralNetwork::predict(const Eigen::VectorXd &input) const
{
    return feedforward(input);
}

void NeuralNetwork::check_input_size(const Eigen::VectorXd &input) const
{
    if (input.size() != layers.front().weights.cols())
    {
        throw TrainingDataError("Input size does not match the first layer input size");
    }
}

void NeuralNetwork::check_target_size(const Eigen::VectorXd &target) const
{
    if (target.size() != layers.back().weights.rows())
    {
        throw TrainingDataError("Target size does not match the output layer size");
    }
}