#include "neural_network.hpp"
#include "neural_network_common.hpp"

Eigen::VectorXd NeuralNetwork::feedforward(const Eigen::VectorXd &input) const {
    if (input.size() != layers.front().weights.cols()) {
        throw SizeMismatchError("Input size does not match the first layer size");
    }

    Eigen::VectorXd activation = input;

    for (const auto& layer : layers) {
        if (activation.size() != layer.weights.cols()) {
            throw SizeMismatchError("Activation size does not match weight matrix dimensions");
        }

        Eigen::VectorXd z = layer.weights * activation + layer.biases;

        if (layer.batch_norm) {
            auto [bn_output, _] = layer.batch_norm->forward(z, true);
            z = bn_output;
        }

        if (!is_valid(z)) {
            throw NumericalInstabilityError("Invalid values detected in layer pre-activation");
        }

        if (&layer == &layers.back()) {
            activation = activation_function.activateOutput(z);
        } else {
            activation = activation_function.activateHidden(z);
        }

        if (!is_valid(activation)) {
            throw NumericalInstabilityError("Invalid values detected in layer activation");
        }
    }

    return activation;
}

std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
NeuralNetwork::feedforward_with_intermediates(const Eigen::VectorXd &input) const {
    if (input.size() != layers.front().weights.cols()) {
        throw SizeMismatchError("Input size does not match the first layer size");
    }

    std::vector<Eigen::VectorXd> activations;
    std::vector<Eigen::VectorXd> z_values;
    activations.push_back(input);

    Eigen::VectorXd activation = input;

    for (const auto& layer : layers) {
        if (activation.size() != layer.weights.cols()) {
            throw SizeMismatchError("Activation size does not match weight matrix dimensions");
        }

        Eigen::VectorXd z = layer.weights * activation + layer.biases;

        if (layer.batch_norm) {
            auto [bn_output, _] = layer.batch_norm->forward(z, true);
            z = bn_output;
        }

        z_values.push_back(z);

        if (!is_valid(z)) {
            throw NumericalInstabilityError("Invalid values detected in layer pre-activation");
        }

        if (&layer == &layers.back()) {
            activation = activation_function.activateOutput(z);
        } else {
            activation = activation_function.activateHidden(z);
        }

        if (!is_valid(activation)) {
            throw NumericalInstabilityError("Invalid values detected in layer activation");
        }

        activations.push_back(activation);
    }

    return {activations, z_values};
}
