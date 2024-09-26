#include "neural_network.hpp"
#include "neural_network_common.hpp"

#include "neural_network.hpp"
#include "neural_network_common.hpp"

Eigen::VectorXd NeuralNetwork::feedforward(const Eigen::VectorXd &input) const {
    if (input.size() != layers.front().weights.cols()) {
        throw SizeMismatchError("Input size does not match the first layer size");
    }

    Eigen::VectorXd activation = input;
    
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        if (activation.size() != layer.weights.cols()) {
            throw SizeMismatchError("Activation size does not match weight matrix dimensions");
        }

        Eigen::VectorXd z = layer.weights * activation + layer.biases;
        
        if (!is_valid(z)) {
            throw NumericalInstabilityError("Invalid values detected in layer " + std::to_string(i) + " pre-activation");
        }

        if (layer.batch_norm) {
            auto [bn_output, _] = layer.batch_norm->forward(z, false);  // Use false for inference
            z = bn_output;
            
            if (!is_valid(z)) {
                throw NumericalInstabilityError("Invalid values detected after batch normalization in layer " + std::to_string(i));
            }
        }

        if (i == layers.size() - 1) {
            activation = activation_function.activateOutput(z);
        } else {
            activation = activation_function.activateHidden(z);
        }

        if (!is_valid(activation)) {
            throw NumericalInstabilityError("Invalid values detected in layer " + std::to_string(i) + " activation");
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

    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        if (activation.size() != layer.weights.cols()) {
            throw SizeMismatchError("Activation size does not match weight matrix dimensions");
        }

        Eigen::VectorXd z = layer.weights * activation + layer.biases;
        
        if (!is_valid(z)) {
            throw NumericalInstabilityError("Invalid values detected in layer " + std::to_string(i) + " pre-activation");
        }

        if (layer.batch_norm) {
            auto [bn_output, _] = layer.batch_norm->forward(z, true);  // Use true for training
            z = bn_output;
            
            if (!is_valid(z)) {
                throw NumericalInstabilityError("Invalid values detected after batch normalization in layer " + std::to_string(i));
            }
        }

        z_values.push_back(z);

        if (i == layers.size() - 1) {
            activation = activation_function.activateOutput(z);
        } else {
            activation = activation_function.activateHidden(z);
        }

        if (!is_valid(activation)) {
            throw NumericalInstabilityError("Invalid values detected in layer " + std::to_string(i) + " activation");
        }

        activations.push_back(activation);
    }

    return {activations, z_values};
}
