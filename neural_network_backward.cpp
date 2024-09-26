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

    // Clip delta to prevent extreme values
    clip_and_check(delta, "Output delta", 1e3);

    // Backpropagate error
    for (int i = layers.size() - 1; i >= 0; --i) {
        weight_gradients[i] = delta * activations[i].transpose();
        bias_gradients[i] = delta;

        // Gradient clipping
        clip_and_check(weight_gradients[i], "Weight gradients", 1e3);
        clip_and_check(bias_gradients[i], "Bias gradients", 1e3);

        if (i > 0) {
            Eigen::VectorXd d_layer = layers[i].weights.transpose() * delta;

            if (layers[i-1].batch_norm) {
                BatchNorm::BatchNormCache cache;
                cache.x = z_values[i-1];
                Eigen::VectorXd mean = z_values[i-1].mean() * Eigen::VectorXd::Ones(z_values[i-1].size());
                Eigen::VectorXd var = ((z_values[i-1].array() - mean.array()).square().mean()) * Eigen::VectorXd::Ones(z_values[i-1].size());
                cache.mean.push_back(mean);
                cache.var.push_back(var);
                cache.normalized.push_back((z_values[i-1].array() - mean.array()) / (var.array() + layers[i-1].batch_norm->get_epsilon()).sqrt());

                d_layer = layers[i-1].batch_norm->backward(d_layer, cache);
                layers[i-1].bn_gamma_grad = layers[i-1].batch_norm->get_gamma_grad();
                layers[i-1].bn_beta_grad = layers[i-1].batch_norm->get_beta_grad();
            }

            delta = d_layer.cwiseProduct(activation_function.derivativeHidden(z_values[i-1]));
            clip_and_check(delta, "Hidden delta", 1e3);
        }
    }

    return {weight_gradients, bias_gradients};
}