#include "neural_network.hpp"
#include "exceptions.hpp"
#include <algorithm>
#include <random>
#include <numeric>
#include <iostream>

NeuralNetwork::NeuralNetwork(const std::vector<int> &layer_sizes,
                             ActivationFunction::Type hidden_activation,
                             ActivationFunction::Type output_activation,
                             WeightInitialization weight_init,
                             const std::string &optimizer_name,
                             double learning_rate,
                             RegularizationType reg_type,
                             double reg_strength)
    : layers(layer_sizes),
      activation_function(hidden_activation, output_activation),
      weight_init(weight_init),
      regularization_type(reg_type),
      regularization_strength(reg_strength)
{
    std::cout << "NeuralNetwork constructor started" << std::endl;

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

    weights.resize(layers.size() - 1);
    biases.resize(layers.size() - 1);

    std::cout << "Creating optimizer" << std::endl;
    optimizer = create_optimizer_for_network(optimizer_name, learning_rate);
    std::cout << "Optimizer created" << std::endl;

    initialize_weights();
    std::cout << "Weights initialized" << std::endl;

    std::cout << "NeuralNetwork constructor completed" << std::endl;
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
    validate();
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

Eigen::VectorXd NeuralNetwork::feedforward(const Eigen::VectorXd &input) const
{
    Eigen::VectorXd activation = input;

    for (size_t i = 0; i < weights.size() - 1; ++i)
    {
        Eigen::VectorXd z = weights[i] * activation + biases[i];
        activation = activation_function.activateHidden(z);
    }

    Eigen::VectorXd z_output = weights.back() * activation + biases.back();
    return activation_function.activateOutput(z_output);
}

std::vector<Eigen::VectorXd> NeuralNetwork::feedforward_with_intermediates(const Eigen::VectorXd &input) const
{
    std::cout << "feedforward_with_intermediates started" << std::endl;
    std::vector<Eigen::VectorXd> activations;
    activations.push_back(input);

    Eigen::VectorXd activation = input;

    for (size_t i = 0; i < weights.size() - 1; ++i)
    {
        std::cout << "Processing layer " << i << std::endl;
        Eigen::VectorXd z = weights[i] * activation + biases[i];
        activation = activation_function.activateHidden(z);
        activations.push_back(activation);
    }

    std::cout << "Processing output layer" << std::endl;
    Eigen::VectorXd z_output = weights.back() * activation + biases.back();
    Eigen::VectorXd output = activation_function.activateOutput(z_output);
    activations.push_back(output);

    std::cout << "feedforward_with_intermediates completed. Activations size: " << activations.size() << std::endl;
    return activations;
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>>
NeuralNetwork::backpropagate(const Eigen::VectorXd &input, const Eigen::VectorXd &target)
{
    std::cout << "backpropagate started" << std::endl;
    std::cout << "Input size: " << input.size() << ", Target size: " << target.size() << std::endl;
    
    check_input_size(input);
    check_target_size(target);

    std::cout << "Sizes checked" << std::endl;

    std::vector<Eigen::VectorXd> activations = feedforward_with_intermediates(input);
    std::cout << "Feedforward completed. Activations size: " << activations.size() << std::endl;

    std::vector<Eigen::VectorXd> zs;
    zs.reserve(weights.size());

    std::cout << "Calculating zs" << std::endl;
    for (size_t i = 0; i < weights.size(); ++i)
    {
        std::cout << "Calculating z for layer " << i << std::endl;
        zs.push_back(weights[i] * activations[i] + biases[i]);
    }
    std::cout << "zs calculated. Size: " << zs.size() << std::endl;

    std::vector<Eigen::VectorXd> deltas(layers.size() - 1);
    std::vector<Eigen::MatrixXd> weight_gradients(weights.size());
    std::vector<Eigen::VectorXd> bias_gradients(biases.size());

    std::cout << "Calculating output layer error" << std::endl;
    // Calculate output layer error
    Eigen::VectorXd output_error;
    if (activation_function.getOutputActivationType() == ActivationFunction::Type::Softmax)
    {
        output_error = activations.back() - target;
    }
    else
    {
        output_error = (activations.back() - target).array() *
                       activation_function.derivativeOutput(zs.back()).array();
    }
    deltas.back() = output_error;

    std::cout << "Backpropagating error" << std::endl;
    // Backpropagate error
    for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i)
    {
        std::cout << "Backpropagating for layer " << i << std::endl;
        std::cout << "weights[" << i+1 << "] shape: " << weights[i+1].rows() << "x" << weights[i+1].cols() << std::endl;
        std::cout << "deltas[" << i+1 << "] size: " << deltas[i+1].size() << std::endl;
        Eigen::VectorXd error = weights[i + 1].transpose() * deltas[i + 1];
        std::cout << "Error calculated" << std::endl;
        std::cout << "zs[" << i << "] size: " << zs[i].size() << std::endl;
        Eigen::VectorXd derivative = activation_function.derivativeHidden(zs[i]);
        std::cout << "Derivative calculated" << std::endl;
        deltas[i] = error.array() * derivative.array();
        std::cout << "Delta calculated for layer " << i << std::endl;
    }

    std::cout << "Computing gradients" << std::endl;
    // Compute gradients
    for (size_t i = 0; i < weights.size(); ++i)
    {
        std::cout << "Computing gradient for layer " << i << std::endl;
        std::cout << "deltas[" << i << "] size: " << deltas[i].size() << std::endl;
        std::cout << "activations[" << i << "] size: " << activations[i].size() << std::endl;
        weight_gradients[i] = deltas[i] * activations[i].transpose();
        bias_gradients[i] = deltas[i];
        std::cout << "Gradient computed for layer " << i << std::endl;
    }

    std::cout << "backpropagate completed" << std::endl;
    return {weight_gradients, bias_gradients};
}

void NeuralNetwork::train(const std::vector<Eigen::VectorXd>& inputs,
                          const std::vector<Eigen::VectorXd>& targets,
                          int epochs,
                          int batch_size,
                          double error_tolerance,
                          double validation_split)
{
    std::cout << "Training started" << std::endl;
    
    if (inputs.empty() || targets.empty()) {
        throw TrainingDataError("Inputs and targets cannot be empty");
    }
    
    check_input_size(inputs[0]);
    check_target_size(targets[0]);

    if (inputs.size() != targets.size())
    {
        throw TrainingDataError("Number of inputs must match number of targets");
    }
    if (epochs <= 0 || batch_size <= 0 || validation_split < 0 || validation_split >= 1)
    {
        throw std::invalid_argument("Invalid training parameters");
    }

    size_t dataset_size = inputs.size();
    size_t validation_size = static_cast<size_t>(dataset_size * validation_split);
    size_t training_size = dataset_size - validation_size;

    std::cout << "Dataset size: " << dataset_size << ", Training size: " << training_size << ", Validation size: " << validation_size << std::endl;

    std::vector<size_t> indices(dataset_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::cout << "Epoch " << epoch + 1 << " started" << std::endl;

        std::shuffle(indices.begin(), indices.end(), g);

        // Training
        for (size_t i = 0; i < training_size; i += batch_size)
        {
            size_t batch_end = std::min(i + batch_size, training_size);
            std::vector<Eigen::VectorXd> batch_inputs, batch_targets;
            for (size_t j = i; j < batch_end; ++j)
            {
                batch_inputs.push_back(inputs[indices[j]]);
                batch_targets.push_back(targets[indices[j]]);
            }
            std::cout << "Updating batch: " << i << " to " << batch_end << std::endl;
            update_batch(batch_inputs, batch_targets);
        }

        std::cout << "Computing losses" << std::endl;
        // Compute losses
        std::vector<Eigen::VectorXd> train_inputs, train_targets, val_inputs, val_targets;
        for (size_t i = 0; i < training_size; ++i)
        {
            train_inputs.push_back(inputs[indices[i]]);
            train_targets.push_back(targets[indices[i]]);
        }
        for (size_t i = training_size; i < dataset_size; ++i)
        {
            val_inputs.push_back(inputs[indices[i]]);
            val_targets.push_back(targets[indices[i]]);
        }

        std::cout << "Calculating train loss" << std::endl;
        double train_loss = get_loss(train_inputs, train_targets);
        std::cout << "Calculating validation loss" << std::endl;
        double val_loss = get_loss(val_inputs, val_targets);

        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << ", Train loss: " << train_loss
                  << ", Validation loss: " << val_loss << std::endl;

        if (train_loss < error_tolerance)
        {
            std::cout << "Reached error tolerance. Stopping training." << std::endl;
            break;
        }
    }

    std::cout << "Training completed" << std::endl;
}

void NeuralNetwork::update_batch(const std::vector<Eigen::VectorXd>& batch_inputs,
                                 const std::vector<Eigen::VectorXd>& batch_targets)
{
    std::cout << "update_batch started with batch size: " << batch_inputs.size() << std::endl;
    
    std::vector<Eigen::MatrixXd> weight_gradients(weights.size());
    std::vector<Eigen::VectorXd> bias_gradients(biases.size());

    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << "Initializing gradient for layer " << i << std::endl;
        weight_gradients[i] = Eigen::MatrixXd::Zero(weights[i].rows(), weights[i].cols());
        bias_gradients[i] = Eigen::VectorXd::Zero(biases[i].size());
    }

    for (size_t i = 0; i < batch_inputs.size(); ++i)
    {
        std::cout << "Processing sample " << i << " in batch" << std::endl;
        std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>> gradients =
            backpropagate(batch_inputs[i], batch_targets[i]);

        std::cout << "Backpropagation completed for sample " << i << std::endl;
        std::vector<Eigen::MatrixXd>& sample_weight_gradients = gradients.first;
        std::vector<Eigen::VectorXd>& sample_bias_gradients = gradients.second;

        for (size_t j = 0; j < weights.size(); ++j)
        {
            std::cout << "Accumulating gradients for layer " << j << std::endl;
            weight_gradients[j] += sample_weight_gradients[j];
            bias_gradients[j] += sample_bias_gradients[j];
        }
        std::cout << "Gradients accumulated for sample " << i << std::endl;
    }

    // Average gradients
    for (size_t i = 0; i < weights.size(); ++i)
    {
        weight_gradients[i] /= batch_inputs.size();
        bias_gradients[i] /= batch_inputs.size();
    }

    std::cout << "Applying regularization" << std::endl;
    apply_regularization(weight_gradients, bias_gradients);

    std::cout << "Updating weights and biases" << std::endl;
    for (size_t i = 0; i < weights.size(); ++i)
    {
        optimizer->update(weights[i], biases[i], weight_gradients[i], bias_gradients[i]);
    }
    std::cout << "update_batch completed" << std::endl;
}

double NeuralNetwork::get_loss(const std::vector<Eigen::VectorXd> &inputs,
                               const std::vector<Eigen::VectorXd> &targets) const
{
    std::cout << "get_loss started" << std::endl;
    if (inputs.size() != targets.size())
    {
        throw TrainingDataError("Number of inputs must match number of targets");
    }

    double total_loss = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        Eigen::VectorXd prediction = predict(inputs[i]);
        if (activation_function.getOutputActivationType() == ActivationFunction::Type::Softmax)
        {
            // Cross-entropy loss for Softmax
            total_loss -= (targets[i].array() * prediction.array().log()).sum();
        }
        else
        {
            // Mean squared error for other activation functions
            total_loss += (prediction - targets[i]).squaredNorm();
        }
    }
    std::cout << "get_loss completed" << std::endl;
    return total_loss / inputs.size();
}

void NeuralNetwork::apply_regularization(std::vector<Eigen::MatrixXd> &weight_gradients,
                                         std::vector<Eigen::VectorXd> &bias_gradients)
{
    std::cout << "apply_regularization started" << std::endl;
    switch (regularization_type)
    {
    case RegularizationType::L1:
        for (size_t i = 0; i < weights.size(); ++i)
        {
            weight_gradients[i].array() += regularization_strength * weights[i].array().sign();
        }
        break;
    case RegularizationType::L2:
        for (size_t i = 0; i < weights.size(); ++i)
        {
            weight_gradients[i].array() += regularization_strength * weights[i].array();
        }
        break;
    default:
        break;
    }
    std::cout << "apply_regularization completed" << std::endl;
}

Eigen::VectorXd NeuralNetwork::predict(const Eigen::VectorXd &input) const
{
    std::cout << "predict started" << std::endl;
    Eigen::VectorXd result = feedforward(input);
    std::cout << "predict completed" << std::endl;
    return result;
}