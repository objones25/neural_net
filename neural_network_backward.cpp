#include "neural_network.hpp"
#include "neural_network_common.hpp"

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>>
NeuralNetwork::backpropagate(const Eigen::Ref<const Eigen::VectorXd> &input,
                             const Eigen::Ref<const Eigen::VectorXd> &target) {
    if (input.size() != layers.front().weights.cols()) {
        throw SizeMismatchError("Input size does not match the first layer size");
    }
    if (target.size() != layers.back().weights.rows()) {
        throw SizeMismatchError("Target size does not match the output layer size");
    }

    auto [activations, z_values] = feedforward_with_intermediates(input);
    
    std::vector<Eigen::MatrixXd> weight_gradients(layers.size());
    std::vector<Eigen::VectorXd> bias_gradients(layers.size());

    // Calculate output layer error
    Eigen::VectorXd delta = activations.back() - target;
    if (activation_function.getOutputActivationType() != ActivationFunction::Type::Softmax) {
        delta = delta.cwiseProduct(activation_function.derivativeOutput(z_values.back()));
    }

    const double safe_epsilon = 1e-12;

    // Backpropagate error
    for (int i = layers.size() - 1; i >= 0; --i) {
        weight_gradients[i] = delta * activations[i].transpose();
        bias_gradients[i] = delta;

        // Gradient clipping
        weight_gradients[i] = weight_gradients[i].cwiseMin(1.0).cwiseMax(-1.0);
        bias_gradients[i] = bias_gradients[i].cwiseMin(1.0).cwiseMax(-1.0);

        if (i > 0) {
            Eigen::VectorXd d_layer = layers[i].weights.transpose() * delta;

            if (layers[i-1].batch_norm) {
                BatchNorm::BatchNormCache cache;
                cache.x = z_values[i-1];
                Eigen::VectorXd mean = z_values[i-1].mean() * Eigen::VectorXd::Ones(z_values[i-1].size());
                Eigen::VectorXd var = ((z_values[i-1].array() - mean.array()).square().mean()) * Eigen::VectorXd::Ones(z_values[i-1].size());
                cache.mean.push_back(mean);
                cache.var.push_back(var);
                cache.normalized.push_back((z_values[i-1].array() - mean.array()) / (var.array() + layers[i-1].batch_norm->get_epsilon() + safe_epsilon).sqrt());

                d_layer = layers[i-1].batch_norm->backward(d_layer, cache);
                layers[i-1].bn_gamma_grad = layers[i-1].batch_norm->get_gamma_grad();
                layers[i-1].bn_beta_grad = layers[i-1].batch_norm->get_beta_grad();
            }

            delta = d_layer.cwiseProduct(activation_function.derivativeHidden(z_values[i-1]));
        }

        // Check for non-finite values
        if (!delta.allFinite()) {
            std::cerr << "Non-finite values detected in delta at layer " << i << std::endl;
            std::cerr << "Delta: " << delta.transpose() << std::endl;
            throw NumericalInstabilityError("Non-finite values detected in delta at layer " + std::to_string(i));
        }

        if (!weight_gradients[i].allFinite() || !bias_gradients[i].allFinite()) {
            std::cerr << "Non-finite values detected in gradients at layer " << i << std::endl;
            throw NumericalInstabilityError("Non-finite values detected in gradients at layer " + std::to_string(i));
        }
    }

    return {weight_gradients, bias_gradients};
}