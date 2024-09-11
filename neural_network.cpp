#include "neural_network.hpp"
#include "exceptions.hpp"
#include <algorithm>
#include <random>
#include <fstream>
#include <numeric>
#include <iostream>

NeuralNetwork::NeuralNetwork(const std::vector<int> &layer_sizes,
                             double lr,
                             ActivationFunction act_func,
                             WeightInitialization weight_init,
                             OptimizationAlgorithm opt_algo,
                             RegularizationType reg_type,
                             double reg_strength)
    : layers(layer_sizes), learning_rate(lr), activation_function(act_func),
      weight_init(weight_init), optimization_algo(opt_algo),
      regularization_type(reg_type), regularization_strength(reg_strength)
{
    if (layer_sizes.size() < 2)
    {
        throw NetworkConfigurationError("Network must have at least two layers");
    }
    for (int size : layer_sizes)
    {
        if (size <= 0)
        {
            throw NetworkConfigurationError("Each layer must have at least one neuron");
        }
    }
    if (lr <= 0 || lr >= 1)
    {
        throw NetworkConfigurationError("Learning rate must be between 0 and 1");
    }

    weights.resize(layers.size() - 1);
    biases.resize(layers.size() - 1);

    if (optimization_algo == OptimizationAlgorithm::Adam ||
        optimization_algo == OptimizationAlgorithm::RMSprop)
    {
        m_weights.resize(layers.size() - 1);
        m_biases.resize(layers.size() - 1);
        if (optimization_algo == OptimizationAlgorithm::Adam)
        {
            v_weights.resize(layers.size() - 1);
            v_biases.resize(layers.size() - 1);
        }
    }

    initialize_weights();
    validate();
}

void NeuralNetwork::validate() const
{
    if (layers.size() < 2)
    {
        throw NetworkConfigurationError("Network must have at least two layers");
    }
    if (weights.size() != layers.size() - 1)
    {
        throw NetworkConfigurationError("Number of weight matrices must match number of layers minus one");
    }
    if (biases.size() != layers.size() - 1)
    {
        throw NetworkConfigurationError("Number of bias vectors must match number of layers minus one");
    }

    for (size_t i = 0; i < weights.size(); ++i)
    {
        if (weights[i].rows() != layers[i + 1] || weights[i].cols() != layers[i])
        {
            throw NetworkConfigurationError("Weight matrix dimensions mismatch");
        }
        if (biases[i].size() != layers[i + 1])
        {
            throw NetworkConfigurationError("Bias vector size mismatch");
        }
    }
}

void NeuralNetwork::initialize_weights()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    for (size_t i = 1; i < layers.size(); ++i)
    {
        int fan_in = layers[i - 1];
        int fan_out = layers[i];

        std::normal_distribution<> d;
        switch (weight_init)
        {
        case WeightInitialization::Random:
            d = std::normal_distribution<>(0.0, 0.1);
            break;
        case WeightInitialization::Xavier:
            d = std::normal_distribution<>(0.0, std::sqrt(2.0 / (fan_in + fan_out)));
            break;
        case WeightInitialization::He:
            d = std::normal_distribution<>(0.0, std::sqrt(2.0 / fan_in));
            break;
        }

        weights[i - 1] = Eigen::MatrixXd::NullaryExpr(layers[i], layers[i - 1],
                                                      [&]()
                                                      { return d(gen); });
        biases[i - 1] = Eigen::VectorXd::NullaryExpr(layers[i],
                                                     [&]()
                                                     { return d(gen); });
    }
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

std::vector<Eigen::VectorXd> NeuralNetwork::feedforward(const Eigen::VectorXd &input) const
{
    check_input_size(input);

    std::vector<Eigen::VectorXd> activations;
    activations.push_back(input);

    for (size_t i = 0; i < weights.size(); ++i)
    {
        Eigen::VectorXd z = weights[i] * activations.back() + biases[i];
        activations.push_back(activate(z, activation_function));
    }

    return activations;
}

void NeuralNetwork::backpropagate(const Eigen::VectorXd &input, const Eigen::VectorXd &target)
{
    check_input_size(input);
    check_target_size(target);

    auto activations = feedforward(input);
    std::vector<Eigen::VectorXd> zs;
    for (size_t i = 0; i < weights.size(); ++i) {
        zs.push_back(weights[i] * activations[i] + biases[i]);
    }

    std::vector<Eigen::VectorXd> deltas(layers.size() - 1);

    // Calculate output layer error
    Eigen::VectorXd output_error;
    if (activation_function == ActivationFunction::Softmax) {
        output_error = activations.back() - target;
    } else {
        output_error = (activations.back() - target).array() * 
                       activate_derivative(zs.back(), activation_function).array();
    }
    deltas.back() = output_error;

    // Backpropagate error
    for (int i = layers.size() - 2; i > 0; --i)
    {
        Eigen::VectorXd error = weights[i].transpose() * deltas[i];
        deltas[i - 1] = error.array() * activate_derivative(zs[i-1], activation_function).array();
    }

    // Update weights and biases
    for (size_t i = 0; i < weights.size(); ++i)
    {
        Eigen::MatrixXd weight_gradient = deltas[i] * activations[i].transpose();
        Eigen::VectorXd bias_gradient = deltas[i];

        // Apply regularization
        switch (regularization_type)
        {
        case RegularizationType::L1:
            weight_gradient.array() += regularization_strength * weights[i].array().sign();
            break;
        case RegularizationType::L2:
            weight_gradient.array() += regularization_strength * weights[i].array();
            break;
        default:
            break;
        }

        // Apply optimization algorithm
        update_weights_and_biases(optimization_algo,
                                  weights[i], biases[i],
                                  weight_gradient, bias_gradient,
                                  m_weights[i], m_biases[i],
                                  v_weights[i], v_biases[i],
                                  learning_rate, beta1, beta2, epsilon,
                                  t);
    }
}

void NeuralNetwork::train(const std::vector<Eigen::VectorXd>& inputs,
                          const std::vector<Eigen::VectorXd>& targets,
                          int epochs,
                          int batch_size,
                          double error_tolerance,
                          double validation_split)
{
    if (inputs.size() != targets.size()) {
        throw TrainingDataError("Number of inputs must match number of targets");
    }
    if (epochs <= 0 || batch_size <= 0 || validation_split < 0 || validation_split >= 1) {
        throw std::invalid_argument("Invalid training parameters");
    }

    // Split data into training and validation sets
    size_t validation_size = static_cast<size_t>(inputs.size() * validation_split);
    size_t training_size = inputs.size() - validation_size;

    std::vector<size_t> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<Eigen::VectorXd> train_inputs, train_targets, val_inputs, val_targets;
    for (size_t i = 0; i < training_size; ++i) {
        train_inputs.push_back(inputs[indices[i]]);
        train_targets.push_back(targets[indices[i]]);
    }
    for (size_t i = training_size; i < inputs.size(); ++i) {
        val_inputs.push_back(inputs[indices[i]]);
        val_targets.push_back(targets[indices[i]]);
    }

    double best_val_loss = std::numeric_limits<double>::max();
    int patience = 10, wait = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle training data
        std::shuffle(indices.begin(), indices.begin() + training_size, g);

        // Mini-batch training
        for (size_t i = 0; i < training_size; i += batch_size) {
            size_t batch_end = std::min(i + batch_size, training_size);
            
            // Process each sample in the batch
            for (size_t j = i; j < batch_end; ++j) {
                backpropagate(train_inputs[indices[j]], train_targets[indices[j]]);
            }
        }

        // Calculate losses
        double train_loss = get_loss(train_inputs, train_targets);
        double val_loss = get_loss(val_inputs, val_targets);

        std::cout << "Epoch " << epoch << ". Training Loss: " << train_loss 
                  << ", Validation Loss: " << val_loss << std::endl;
        // Early stopping
        if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
            wait = 0;
        } else {
            wait++;
            if (wait >= patience) {
                std::cout << "Early stopping at epoch " << epoch << std::endl;
                break;
            }
        }

        // Learning rate decay
        if (epoch % 10 == 0 && epoch > 0) {
            learning_rate *= 0.9;
        }

        if (train_loss < error_tolerance) {
            std::cout << "Reached error tolerance at epoch " << epoch << std::endl;
            break;
        }
    }
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>>NeuralNetwork::compute_gradients(const Eigen::VectorXd& input, const Eigen::VectorXd& target)
{
    std::vector<Eigen::VectorXd> activations = feedforward(input);
    std::vector<Eigen::VectorXd> zs;
    for (size_t i = 0; i < weights.size(); ++i) {
        zs.push_back(weights[i] * activations[i] + biases[i]);
    }

    std::vector<Eigen::VectorXd> deltas(layers.size() - 1);
    std::vector<Eigen::MatrixXd> weight_gradients(weights.size());
    std::vector<Eigen::VectorXd> bias_gradients(biases.size());

    // Compute output layer error
    Eigen::VectorXd output_error;
    if (activation_function == ActivationFunction::Softmax) {
        output_error = activations.back() - target;
    } else {
        output_error = (activations.back() - target).array() * 
                       activate_derivative(zs.back(), activation_function).array();
    }
    deltas.back() = output_error;

    // Backpropagate error
    for (int i = static_cast<int>(layers.size()) - 2; i > 0; --i) {
        Eigen::VectorXd error = weights[i].transpose() * deltas[i];
        deltas[i-1] = error.array() * activate_derivative(zs[i-1], activation_function).array();
    }

    // Compute gradients
    for (size_t i = 0; i < weights.size(); ++i) {
        weight_gradients[i] = deltas[i] * activations[i].transpose();
        bias_gradients[i] = deltas[i];
    }

    return {weight_gradients, bias_gradients};
}

void NeuralNetwork::apply_regularization(std::vector<Eigen::MatrixXd>& weight_gradients,
                                         std::vector<Eigen::VectorXd>& bias_gradients)
{
    switch (regularization_type) {
        case RegularizationType::L1:
            for (size_t i = 0; i < weights.size(); ++i) {
                weight_gradients[i].array() += regularization_strength * weights[i].array().sign();
                bias_gradients[i].array() += regularization_strength * biases[i].array().sign();
            }
            break;
        case RegularizationType::L2:
            for (size_t i = 0; i < weights.size(); ++i) {
                weight_gradients[i].array() += regularization_strength * weights[i].array();
                bias_gradients[i].array() += regularization_strength * biases[i].array();
            }
            break;
        default:
            break;
    }
}

Eigen::VectorXd NeuralNetwork::predict(const Eigen::VectorXd &input) const
{
    auto activations = feedforward(input);
    return activations.back();
}

Eigen::VectorXd NeuralNetwork::predict_with_softmax(const Eigen::VectorXd &input) const
{
    Eigen::VectorXd output = predict(input);
    Eigen::VectorXd exp_output = output.array().exp();
    return exp_output / exp_output.sum();
}

double NeuralNetwork::get_loss(const std::vector<Eigen::VectorXd> &inputs,
                               const std::vector<Eigen::VectorXd> &targets) const
{
    if (inputs.size() != targets.size())
    {
        throw TrainingDataError("Number of inputs must match number of targets");
    }

    double total_loss = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        Eigen::VectorXd prediction = predict(inputs[i]);
        Eigen::VectorXd error = prediction - targets[i];
        total_loss += error.squaredNorm();
    }
    return total_loss / (2 * inputs.size());
}

void NeuralNetwork::save_weights(const std::string &filename) const
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Unable to open file for writing");
    }

    for (const auto &w : weights)
    {
        Eigen::MatrixXf w_float = w.cast<float>();
        file.write(reinterpret_cast<char *>(w_float.data()), w_float.size() * sizeof(float));
    }

    for (const auto &b : biases)
    {
        Eigen::VectorXf b_float = b.cast<float>();
        file.write(reinterpret_cast<char *>(b_float.data()), b_float.size() * sizeof(float));
    }
}

void NeuralNetwork::load_weights(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Unable to open file for reading");
    }

    for (auto &w : weights)
    {
        Eigen::MatrixXf w_float(w.rows(), w.cols());
        file.read(reinterpret_cast<char *>(w_float.data()), w_float.size() * sizeof(float));
        w = w_float.cast<double>();
    }

    for (auto &b : biases)
    {
        Eigen::VectorXf b_float(b.size());
        file.read(reinterpret_cast<char *>(b_float.data()), b_float.size() * sizeof(float));
        b = b_float.cast<double>();
    }

    validate();
}