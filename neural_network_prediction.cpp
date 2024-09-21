#include "neural_network.hpp"
#include "neural_network_common.hpp"

Eigen::VectorXd NeuralNetwork::predict(const Eigen::VectorXd &input) const
{
    return feedforward(input);
}

void NeuralNetwork::check_input_size(const Eigen::VectorXd &input) const
{
    if (input.size() != layers.front())
    {
        throw TrainingDataError("Input size does not match the first layer size");
    }
}

void NeuralNetwork::check_target_size(const Eigen::VectorXd &target) const
{
    if (target.size() != layers.back())
    {
        throw TrainingDataError("Target size does not match the output layer size");
    }
}