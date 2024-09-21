#include "neural_network.hpp"
#include "neural_network_common.hpp"
#include "exceptions.hpp"

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>>
NeuralNetwork::backpropagate(const Eigen::VectorXd &input, const Eigen::VectorXd &target)
{
    if (input.size() != layers.front()) {
        throw SizeMismatchError("Input size does not match the first layer size");
    }
    if (target.size() != layers.back()) {
        throw SizeMismatchError("Target size does not match the output layer size");
    }

    auto [activations, z_values] = feedforward_with_intermediates(input);
    std::vector<Eigen::VectorXd> deltas(layers.size() - 1);
    std::vector<Eigen::MatrixXd> weight_gradients(weights.size());
    std::vector<Eigen::VectorXd> bias_gradients(biases.size());
    std::vector<Eigen::VectorXd> bn_gamma_gradients(batch_norms.size());
    std::vector<Eigen::VectorXd> bn_beta_gradients(batch_norms.size());

    // Initialize gradients
    for (size_t i = 0; i < weights.size(); ++i)
    {
        weight_gradients[i] = Eigen::MatrixXd::Zero(weights[i].rows(), weights[i].cols());
        bias_gradients[i] = Eigen::VectorXd::Zero(biases[i].size());
    }
    for (size_t i = 0; i < batch_norms.size(); ++i)
    {
        bn_gamma_gradients[i] = Eigen::VectorXd::Zero(layers[i + 1]);
        bn_beta_gradients[i] = Eigen::VectorXd::Zero(layers[i + 1]);
    }

    // Calculate output layer error
    Eigen::VectorXd output_error;
    if (activation_function.getOutputActivationType() == ActivationFunction::Type::Softmax)
    {
        output_error = activations.back() - target;
    }
    else
    {
        output_error = (activations.back() - target).array() *
                       activation_function.derivativeOutput(z_values.back()).array();
    }
    deltas.back() = output_error;

    // Backpropagate error
    for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i)
    {
        if (deltas[i + 1].size() != batch_norms[i].get_features()) {
            throw SizeMismatchError("Delta size mismatch for batch normalization at layer " + std::to_string(i));
        }

        Eigen::VectorXd d_batch_norm;
        try {
            d_batch_norm = batch_norms[i].backward(deltas[i + 1], z_values[i]);
        } catch (const std::exception& e) {
            throw NumericalInstabilityError("Batch normalization backward pass failed at layer " + std::to_string(i) + ": " + e.what());
        }

        if (d_batch_norm.size() != weights[i].rows()) {
            throw SizeMismatchError("Batch norm gradient size does not match weight matrix dimensions at layer " + std::to_string(i));
        }

        Eigen::VectorXd error = weights[i].transpose() * d_batch_norm;
        Eigen::VectorXd derivative = activation_function.derivativeHidden(z_values[i]);
        deltas[i] = error.array() * derivative.array();

        // Calculate batch norm gradients
        bn_gamma_gradients[i] = (deltas[i + 1].array() * z_values[i].array()).matrix();
        bn_beta_gradients[i] = deltas[i + 1];
    }

    // Compute gradients
    for (size_t i = 0; i < weights.size(); ++i)
    {
        if (deltas[i].size() != weights[i].rows() || activations[i].size() != weights[i].cols()) {
            throw SizeMismatchError("Delta or activation size mismatch for weight gradient calculation at layer " + std::to_string(i));
        }

        weight_gradients[i] = deltas[i] * activations[i].transpose();
        bias_gradients[i] = deltas[i];

        // Check for valid gradients
        if (!is_valid(weight_gradients[i]) || !is_valid(bias_gradients[i]))
        {
            throw NumericalInstabilityError("Invalid gradients detected for layer " + std::to_string(i));
        }
    }

    // Clip gradients
    const double clip_value = 1.0;
    for (auto &grad : weight_gradients)
    {
        grad = grad.array().unaryExpr([clip_value](double x)
                                      { return std::max(std::min(x, clip_value), -clip_value); });
    }
    for (auto &grad : bias_gradients)
    {
        grad = grad.array().unaryExpr([clip_value](double x)
                                      { return std::max(std::min(x, clip_value), -clip_value); });
    }

    return {weight_gradients, bias_gradients};
}