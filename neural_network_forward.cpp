#include "neural_network.hpp"
#include "neural_network_common.hpp"

Eigen::VectorXd NeuralNetwork::feedforward(const Eigen::VectorXd &input) const
{
    Eigen::VectorXd activation = input;
    // Propagate through hidden layers
    for (size_t i = 0; i < weights.size() - 1; ++i)
    {
        Eigen::VectorXd z = weights[i] * activation + biases[i];
        z = batch_norms[i].forward(z, true); // Apply batch normalization
        if (!is_valid(z))
        {
            throw NumericalInstabilityError("Invalid values detected in layer " + std::to_string(i) + " pre-activation");
        }
        activation = activation_function.activateHidden(z);
        if (!is_valid(activation))
        {
            throw NumericalInstabilityError("Invalid values detected in layer " + std::to_string(i) + " activation");
        }
    }

    // Compute output layer
    Eigen::VectorXd z_output = weights.back() * activation + biases.back();
    // Note: typically we don't apply batch norm to the output layer
    if (!is_valid(z_output))
    {
        throw NumericalInstabilityError("Invalid values detected in output layer pre-activation");
    }
    Eigen::VectorXd output = activation_function.activateOutput(z_output);
    if (!is_valid(output))
    {
        throw NumericalInstabilityError("Invalid values detected in output layer activation");
    }
    return output;
}

std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
NeuralNetwork::feedforward_with_intermediates(const Eigen::VectorXd &input) const
{
    std::vector<Eigen::VectorXd> activations;
    std::vector<Eigen::VectorXd> z_values;
    activations.push_back(input);

    Eigen::VectorXd activation = input;

    // Propagate through hidden layers
    for (size_t i = 0; i < weights.size() - 1; ++i)
    {
        Eigen::VectorXd z = weights[i] * activation + biases[i];
        z = batch_norms[i].forward(z, true); // Apply batch normalization
        z_values.push_back(z);
        if (!is_valid(z))
        {
            throw NumericalInstabilityError("Invalid values detected in layer " + std::to_string(i) + " pre-activation");
        }
        activation = activation_function.activateHidden(z);
        if (!is_valid(activation))
        {
            throw NumericalInstabilityError("Invalid values detected in layer " + std::to_string(i) + " activation");
        }
        activations.push_back(activation);
    }

    // Compute output layer
    Eigen::VectorXd z_output = weights.back() * activation + biases.back();
    z_values.push_back(z_output);
    if (!is_valid(z_output))
    {
        throw NumericalInstabilityError("Invalid values detected in output layer pre-activation");
    }
    Eigen::VectorXd output = activation_function.activateOutput(z_output);
    if (!is_valid(output))
    {
        throw NumericalInstabilityError("Invalid values detected in output layer activation");
    }
    activations.push_back(output);

    return {activations, z_values};
}