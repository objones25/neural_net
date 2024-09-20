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

    weights.resize(layers.size() - 1);
    biases.resize(layers.size() - 1);

    for (size_t i = 0; i < layers.size() - 1; ++i)
    {
        int fan_in = layers[i];
        int fan_out = layers[i + 1];

        std::normal_distribution<> d;
        switch (weight_init)
        {
        case WeightInitialization::Random:
            d = std::normal_distribution<>(0.0, 0.05); // Changed from 0.1 to 0.05
            break;
        case WeightInitialization::Xavier:
            d = std::normal_distribution<>(0.0, std::sqrt(2.0 / (fan_in + fan_out)));
            break;
        case WeightInitialization::He:
            d = std::normal_distribution<>(0.0, std::sqrt(2.0 / fan_in));
            break;
        }

        weights[i] = Eigen::MatrixXd::NullaryExpr(layers[i + 1], layers[i],
                                                  [&]()
                                                  { return d(gen); });
        biases[i] = Eigen::VectorXd::NullaryExpr(layers[i + 1],
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

Eigen::VectorXd NeuralNetwork::feedforward(const Eigen::VectorXd &input) const
{
    Eigen::VectorXd activation = input;
    for (size_t i = 0; i < weights.size() - 1; ++i)
    {
        Eigen::VectorXd z = weights[i] * activation + biases[i];
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

    Eigen::VectorXd z_output = weights.back() * activation + biases.back();
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

std::vector<Eigen::VectorXd> NeuralNetwork::feedforward_with_intermediates(const Eigen::VectorXd &input) const
{
    std::vector<Eigen::VectorXd> activations;
    activations.push_back(input);

    Eigen::VectorXd activation = input;

    for (size_t i = 0; i < weights.size() - 1; ++i)
    {
        Eigen::VectorXd z = weights[i] * activation + biases[i];
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

    Eigen::VectorXd z_output = weights.back() * activation + biases.back();
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

    return activations;
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>>
NeuralNetwork::backpropagate(const Eigen::VectorXd &input, const Eigen::VectorXd &target)
{
    std::vector<Eigen::VectorXd> activations = feedforward_with_intermediates(input);
    std::vector<Eigen::VectorXd> deltas(layers.size() - 1);
    std::vector<Eigen::MatrixXd> weight_gradients(weights.size());
    std::vector<Eigen::VectorXd> bias_gradients(biases.size());

    // Calculate output layer error
    Eigen::VectorXd output_error;
    if (activation_function.getOutputActivationType() == ActivationFunction::Type::Softmax)
    {
        output_error = activations.back() - target;
    }
    else
    {
        output_error = (activations.back() - target).array() *
                       activation_function.derivativeOutput(activations.back()).array();
    }
    deltas.back() = output_error;

    // Backpropagate error
    for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i)
    {
        Eigen::VectorXd error = weights[i].transpose() * deltas[i];
        Eigen::VectorXd derivative = activation_function.derivativeHidden(activations[i + 1]);
        deltas[i] = error.array() * derivative.array();
    }

    // Compute gradients
    for (size_t i = 0; i < weights.size(); ++i)
    {
        weight_gradients[i] = deltas[i] * activations[i].transpose();
        bias_gradients[i] = deltas[i];

        if (!is_valid(weight_gradients[i]) || !is_valid(bias_gradients[i]))
        {
            throw GradientExplodingError("Invalid gradients detected for layer " + std::to_string(i));
        }
    }

    const double clip_value = 5.0;
    for (auto &grad : weight_gradients)
    {
        grad = grad.array().min(clip_value).max(-clip_value);
    }
    for (auto &grad : bias_gradients)
    {
        grad = grad.array().min(clip_value).max(-clip_value);
    }

    return {weight_gradients, bias_gradients};
}

void NeuralNetwork::train(const std::vector<Eigen::VectorXd> &inputs,
                          const std::vector<Eigen::VectorXd> &targets,
                          int epochs,
                          int batch_size,
                          double error_tolerance)
{
    if (inputs.empty() || targets.empty())
    {
        throw TrainingDataError("Inputs and targets cannot be empty");
    }

    check_input_size(inputs[0]);
    check_target_size(targets[0]);

    if (inputs.size() != targets.size())
    {
        throw TrainingDataError("Number of inputs must match number of targets");
    }
    if (epochs <= 0 || batch_size <= 0)
    {
        throw std::invalid_argument("Invalid training parameters");
    }

    size_t dataset_size = inputs.size();

    std::vector<size_t> indices(dataset_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::shuffle(indices.begin(), indices.end(), g);

        // Training
        for (size_t i = 0; i < dataset_size; i += batch_size)
        {
            size_t batch_end = std::min(i + batch_size, dataset_size);
            std::vector<Eigen::VectorXd> batch_inputs, batch_targets;
            for (size_t j = i; j < batch_end; ++j)
            {
                batch_inputs.push_back(inputs[indices[j]]);
                batch_targets.push_back(targets[indices[j]]);
            }
            update_batch(batch_inputs, batch_targets);
        }

        // Compute loss
        double train_loss = get_loss(inputs, targets);

        // Print progress every 5 epochs or on the last epoch
        if (epoch % 5 == 0 || epoch == epochs - 1)
        {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs
                      << ", Loss: " << train_loss << std::endl;
        }

        if (train_loss < error_tolerance)
        {
            std::cout << "Reached error tolerance. Stopping training." << std::endl;
            break;
        }
    }
}

void NeuralNetwork::update_batch(const std::vector<Eigen::VectorXd> &batch_inputs,
                                 const std::vector<Eigen::VectorXd> &batch_targets)
{
    std::vector<Eigen::MatrixXd> weight_gradients(weights.size());
    std::vector<Eigen::VectorXd> bias_gradients(biases.size());

    for (size_t i = 0; i < weights.size(); ++i)
    {
        weight_gradients[i] = Eigen::MatrixXd::Zero(weights[i].rows(), weights[i].cols());
        bias_gradients[i] = Eigen::VectorXd::Zero(biases[i].size());
    }

    for (size_t i = 0; i < batch_inputs.size(); ++i)
    {
        std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>> gradients =
            backpropagate(batch_inputs[i], batch_targets[i]);

        std::vector<Eigen::MatrixXd> &sample_weight_gradients = gradients.first;
        std::vector<Eigen::VectorXd> &sample_bias_gradients = gradients.second;

        for (size_t j = 0; j < weights.size(); ++j)
        {
            weight_gradients[j] += sample_weight_gradients[j];
            bias_gradients[j] += sample_bias_gradients[j];
        }
    }

    // Average gradients
    for (size_t i = 0; i < weights.size(); ++i)
    {
        weight_gradients[i] /= batch_inputs.size();
        bias_gradients[i] /= batch_inputs.size();
    }

    apply_regularization(weight_gradients, bias_gradients);

    for (size_t i = 0; i < weights.size(); ++i) {
        optimizer->update(weights[i], biases[i], weight_gradients[i], bias_gradients[i]);
        
        // Replace any NaN or Inf values with 0
        weights[i] = weights[i].unaryExpr([](double x) { return std::isfinite(x) ? x : 0.0; });
        biases[i] = biases[i].unaryExpr([](double x) { return std::isfinite(x) ? x : 0.0; });
    }
}

double NeuralNetwork::get_loss(const std::vector<Eigen::VectorXd> &inputs,
                               const std::vector<Eigen::VectorXd> &targets) const
{
    if (inputs.size() != targets.size())
    {
        throw TrainingDataError("Number of inputs must match number of targets");
    }

    double total_loss = 0.0;
    const double epsilon = 1e-7;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        Eigen::VectorXd prediction = predict(inputs[i]);
        if (activation_function.getOutputActivationType() == ActivationFunction::Type::Softmax)
        {
            total_loss -= (targets[i].array() * (prediction.array() + epsilon).log()).sum();
        }
        else
        {
            total_loss += (prediction - targets[i]).squaredNorm();
        }
    }
    return total_loss / inputs.size();
}

void NeuralNetwork::apply_regularization(std::vector<Eigen::MatrixXd> &weight_gradients,
                                         std::vector<Eigen::VectorXd> &bias_gradients)
{
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
}

Eigen::VectorXd NeuralNetwork::predict(const Eigen::VectorXd &input) const
{
    return feedforward(input);
}

bool NeuralNetwork::is_valid(const Eigen::MatrixXd &mat) const
{
    return ((mat.array() == mat.array()).all() && (mat.array().abs() != std::numeric_limits<double>::infinity()).all());
}

bool NeuralNetwork::is_valid(const Eigen::VectorXd &vec) const
{
    return ((vec.array() == vec.array()).all() && (vec.array().abs() != std::numeric_limits<double>::infinity()).all());
}

void NeuralNetwork::reset()
{
    initialize_weights();
    // Reset any other necessary state
}