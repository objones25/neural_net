#include "neural_network.hpp"
#include "neural_network_common.hpp"
#include "exceptions.hpp"

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>>
NeuralNetwork::backpropagate(const Eigen::Ref<const Eigen::VectorXd>& input, 
                             const Eigen::Ref<const Eigen::VectorXd>& target)
{
    if (input.size() != layers.front())
    {
        throw SizeMismatchError("Input size does not match the first layer size");
    }
    if (target.size() != layers.back())
    {
        throw SizeMismatchError("Target size does not match the output layer size");
    }

    auto [activations, z_values] = feedforward_with_intermediates(input);
    const size_t num_layers = layers.size();
    std::vector<Eigen::VectorXd> deltas(num_layers - 1);
    std::vector<Eigen::MatrixXd> weight_gradients(weights.size());
    std::vector<Eigen::VectorXd> bias_gradients(biases.size());
    std::vector<Eigen::VectorXd> bn_gamma_gradients(batch_norms.size());
    std::vector<Eigen::VectorXd> bn_beta_gradients(batch_norms.size());

    // Initialize gradients
    for (size_t i = 0; i < weights.size(); ++i)
    {
        weight_gradients[i].resize(weights[i].rows(), weights[i].cols());
        weight_gradients[i].setZero();
        bias_gradients[i].resize(biases[i].size());
        bias_gradients[i].setZero();
    }
    for (size_t i = 0; i < batch_norms.size(); ++i)
    {
        bn_gamma_gradients[i].resize(layers[i + 1]);
        bn_gamma_gradients[i].setZero();
        bn_beta_gradients[i].resize(layers[i + 1]);
        bn_beta_gradients[i].setZero();
    }

    // Calculate output layer error
    Eigen::VectorXd& output_error = deltas.back();
    if (activation_function.getOutputActivationType() == ActivationFunction::Type::Softmax)
    {
        output_error = activations.back() - target;
    }
    else
    {
        if (z_values.empty())
        {
            throw std::runtime_error("z_values is empty in backpropagate");
        }
        if (activations.back().size() != target.size() ||
            activations.back().size() != z_values.back().size())
        {
            throw SizeMismatchError("Size mismatch in output error calculation");
        }

        Eigen::VectorXd diff = activations.back() - target;
        Eigen::VectorXd derivative = activation_function.derivativeOutput(z_values.back());

        if (diff.size() != derivative.size())
        {
            throw SizeMismatchError("Size mismatch in output error calculation");
        }

        output_error = diff.cwiseProduct(derivative);

        if (!is_valid(output_error))
        {
            throw NumericalInstabilityError("Invalid values in output error calculation");
        }
    }

    // Backpropagate error
    for (int i = static_cast<int>(num_layers) - 2; i >= 0; --i)
    {
        if (deltas[i + 1].size() != batch_norms[i].get_features())
        {
            throw SizeMismatchError("Delta size mismatch for batch normalization at layer " + std::to_string(i));
        }

        Eigen::VectorXd d_batch_norm;
        try
        {
            d_batch_norm = batch_norms[i].backward(deltas[i + 1], z_values[i]);
        }
        catch (const std::exception& e)
        {
            throw NumericalInstabilityError("Batch normalization backward pass failed at layer " + std::to_string(i) + ": " + e.what());
        }

        if (d_batch_norm.size() != weights[i].rows())
        {
            throw SizeMismatchError("Batch norm gradient size does not match weight matrix dimensions at layer " + std::to_string(i));
        }

        Eigen::VectorXd error = weights[i].transpose() * d_batch_norm;
        Eigen::VectorXd derivative = activation_function.derivativeHidden(z_values[i]);
        deltas[i] = error.cwiseProduct(derivative);

        // Calculate batch norm gradients
        bn_gamma_gradients[i] = deltas[i + 1].cwiseProduct(z_values[i]);
        bn_beta_gradients[i] = deltas[i + 1];
    }

    // Compute gradients
    for (size_t i = 0; i < weights.size(); ++i)
    {
        if (deltas[i].size() != weights[i].rows() || activations[i].size() != weights[i].cols())
        {
            throw SizeMismatchError("Delta or activation size mismatch for weight gradient calculation at layer " + std::to_string(i));
        }

        weight_gradients[i].noalias() = deltas[i] * activations[i].transpose();
        bias_gradients[i] = deltas[i];

        // Check for valid gradients
        if (!is_valid(weight_gradients[i]) || !is_valid(bias_gradients[i]))
        {
            throw NumericalInstabilityError("Invalid gradients detected for layer " + std::to_string(i));
        }
    }

    // Clip gradients
    const double clip_value = 1.0;
    for (auto& grad : weight_gradients)
    {
        grad = grad.cwiseMax(-clip_value).cwiseMin(clip_value);
    }
    for (auto& grad : bias_gradients)
    {
        grad = grad.cwiseMax(-clip_value).cwiseMin(clip_value);
    }

    return {std::move(weight_gradients), std::move(bias_gradients)};
}