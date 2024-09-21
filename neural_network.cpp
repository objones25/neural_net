#include "neural_network.hpp"
#include "exceptions.hpp"
#include <algorithm>
#include <random>
#include <numeric>
#include <iostream>

#define DEBUG_LOG(x) std::cout << "[DEBUG] " << x << std::endl


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

    // Validate network configuration
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

    // Initialize batch_norms vector
    batch_norms.reserve(layers.size() - 1);
    for (size_t i = 1; i < layers.size(); ++i)
    {
        batch_norms.emplace_back(layers[i]);
    }

    // Initialize weights and biases vectors
    weights.resize(layers.size() - 1);
    biases.resize(layers.size() - 1);

    std::cout << "Creating optimizer" << std::endl;
    // Use a slightly larger learning rate
    double adjusted_learning_rate = learning_rate * 10.0; // Increased from 1.0 to 10.0
    optimizer = create_optimizer_for_network(optimizer_name, adjusted_learning_rate);
    std::cout << "Optimizer created with adjusted learning rate: " << adjusted_learning_rate << std::endl;

    // Initialize weights and biases
    initialize_weights();
    std::cout << "Weights initialized" << std::endl;

    std::cout << "NeuralNetwork constructor completed" << std::endl;
}

// Validate the network configuration
void NeuralNetwork::validate() const
{
    // Check for minimum number of layers
    if (layers.size() < 2)
    {
        throw NetworkConfigurationError("Network must have at least two layers");
    }
    // Check if number of weight matrices matches number of layers minus one
    if (weights.size() != layers.size() - 1)
    {
        throw NetworkConfigurationError("Number of weight matrices must match number of layers minus one");
    }
    // Check if number of bias vectors matches number of layers minus one
    if (biases.size() != layers.size() - 1)
    {
        throw NetworkConfigurationError("Number of bias vectors must match number of layers minus one");
    }

    // Check dimensions of weight matrices and bias vectors
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

// Initialize weights and biases
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
        // Choose initialization method based on weight_init
        switch (weight_init)
        {
        case WeightInitialization::Random:
            d = std::normal_distribution<>(0.0, 0.05); // Reduced from 0.1 to 0.05
            break;
        case WeightInitialization::Xavier:
            d = std::normal_distribution<>(0.0, std::sqrt(1.0 / (fan_in + fan_out)));
            break;
        case WeightInitialization::He:
            d = std::normal_distribution<>(0.0, std::sqrt(1.0 / fan_in));
            break;
        }

        // Initialize weights and biases using the chosen distribution
        weights[i] = Eigen::MatrixXd::NullaryExpr(layers[i + 1], layers[i],
                                                  [&]()
                                                  { return d(gen); });
        biases[i] = Eigen::VectorXd::NullaryExpr(layers[i + 1],
                                                 [&]()
                                                 { return d(gen); });
    }

    batch_norms.clear();
    for (size_t i = 0; i < layers.size() - 1; ++i)
    {
        batch_norms.emplace_back(layers[i + 1]);
    }
}

// Check if input size matches the first layer size
void NeuralNetwork::check_input_size(const Eigen::VectorXd &input) const
{
    if (input.size() != layers.front())
    {
        throw TrainingDataError("Input size does not match the first layer size");
    }
}

// Check if target size matches the output layer size
void NeuralNetwork::check_target_size(const Eigen::VectorXd &target) const
{
    if (target.size() != layers.back())
    {
        throw TrainingDataError("Target size does not match the output layer size");
    }
}

// Perform forward propagation through the network
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

// Perform forward propagation and return intermediate activations
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

// Perform backpropagation to compute gradients
std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>>
NeuralNetwork::backpropagate(const Eigen::VectorXd &input, const Eigen::VectorXd &target)
{
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
        Eigen::VectorXd d_batch_norm = batch_norms[i].backward(deltas[i + 1], z_values[i]);
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
        weight_gradients[i] = deltas[i] * activations[i].transpose();
        bias_gradients[i] = deltas[i];

        // Check for valid gradients
        if (!is_valid(weight_gradients[i]) || !is_valid(bias_gradients[i]))
        {
            std::cout << "Invalid gradients detected for layer " << i << std::endl;
            // Replace invalid gradients with zeros
            weight_gradients[i] = Eigen::MatrixXd::Zero(weight_gradients[i].rows(), weight_gradients[i].cols());
            bias_gradients[i] = Eigen::VectorXd::Zero(bias_gradients[i].size());
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

// Train the neural network
void NeuralNetwork::train(const std::vector<Eigen::VectorXd>& inputs,
                          const std::vector<Eigen::VectorXd>& targets,
                          int epochs,
                          int batch_size,
                          double error_tolerance)
{
    try {
        DEBUG_LOG("Starting training");
        if (inputs.empty() || targets.empty()) {
            throw std::invalid_argument("Inputs and targets cannot be empty");
        }
        if (inputs.size() != targets.size()) {
            throw std::invalid_argument("Number of inputs must match number of targets");
        }
        if (epochs <= 0 || batch_size <= 0) {
            throw std::invalid_argument("Invalid training parameters");
        }

        std::cout << "Starting training with " << epochs << " epochs and batch size " << batch_size << std::endl;

        std::vector<size_t> indices(inputs.size());
        std::iota(indices.begin(), indices.end(), 0);

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            DEBUG_LOG("Epoch " << epoch + 1 << "/" << epochs);
            std::shuffle(indices.begin(), indices.end(), generator);

            for (size_t i = 0; i < inputs.size(); i += batch_size) {
                size_t batch_end = std::min(i + batch_size, inputs.size());
                std::vector<Eigen::VectorXd> batch_inputs, batch_targets;

                for (size_t j = i; j < batch_end; ++j) {
                    batch_inputs.push_back(inputs[indices[j]]);
                    batch_targets.push_back(targets[indices[j]]);
                }

                try {
                    update_batch(batch_inputs, batch_targets);
                } catch (const std::exception& e) {
                    std::cerr << "Error in update_batch: " << e.what() << std::endl;
                    throw;
                }
            }

            double loss = get_loss(inputs, targets);
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " Loss: " << loss << std::endl;

            if (loss < error_tolerance) {
                std::cout << "Reached error tolerance. Stopping training." << std::endl;
                break;
            }
        }
        DEBUG_LOG("Training completed");
    } catch (const std::exception& e) {
        std::cerr << "Error in train method: " << e.what() << std::endl;
        throw;
    }
}

void NeuralNetwork::update_batch(const std::vector<Eigen::VectorXd>& batch_inputs,
                                 const std::vector<Eigen::VectorXd>& batch_targets)
{
    try {
        DEBUG_LOG("Starting batch update");
        std::vector<Eigen::MatrixXd> weight_gradients(weights.size());
        std::vector<Eigen::VectorXd> bias_gradients(biases.size());

        for (size_t i = 0; i < weights.size(); ++i) {
            weight_gradients[i] = Eigen::MatrixXd::Zero(weights[i].rows(), weights[i].cols());
            bias_gradients[i] = Eigen::VectorXd::Zero(biases[i].size());
        }

        std::vector<Eigen::MatrixXd> initial_weights = weights;

        for (size_t i = 0; i < batch_inputs.size(); ++i) {
            try {
                std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>> gradients =
                    backpropagate(batch_inputs[i], batch_targets[i]);

                std::vector<Eigen::MatrixXd>& sample_weight_gradients = gradients.first;
                std::vector<Eigen::VectorXd>& sample_bias_gradients = gradients.second;

                for (size_t j = 0; j < weights.size(); ++j) {
                    weight_gradients[j] += sample_weight_gradients[j];
                    bias_gradients[j] += sample_bias_gradients[j];
                }
            } catch (const std::exception& e) {
                std::cerr << "Error in backpropagate: " << e.what() << std::endl;
                throw;
            }
        }

        for (size_t i = 0; i < weights.size(); ++i) {
            weight_gradients[i] /= batch_inputs.size();
            bias_gradients[i] /= batch_inputs.size();
        }

        apply_regularization(weight_gradients, bias_gradients);

        DEBUG_LOG("Weight gradients norm: ");
        for (const auto& grad : weight_gradients) {
            std::cout << grad.norm() << " ";
        }
        std::cout << std::endl;

        for (size_t i = 0; i < weights.size(); ++i) {
            optimizer->update(weights[i], biases[i], weight_gradients[i], bias_gradients[i]);

            DEBUG_LOG("Weight update norm: " << (weights[i] - initial_weights[i]).norm());

            if (i < batch_norms.size()) {
                batch_norms[i].update_parameters(weight_gradients[i], bias_gradients[i], optimizer->get_learning_rate());

                weights[i] = weights[i].unaryExpr([](double x) { return std::isfinite(x) ? x : 0.0; });
                biases[i] = biases[i].unaryExpr([](double x) { return std::isfinite(x) ? x : 0.0; });
            }
        }
        DEBUG_LOG("Batch update completed");
    } catch (const std::exception& e) {
        std::cerr << "Error in update_batch method: " << e.what() << std::endl;
        throw;
    }
}
    
    // Calculate the loss for the given inputs and targets
double NeuralNetwork::get_loss(const std::vector<Eigen::VectorXd>& inputs,
                               const std::vector<Eigen::VectorXd>& targets) const
{
    try {
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
                // Cross-entropy loss for softmax
                total_loss -= (targets[i].array() * (prediction.array() + epsilon).log()).sum();
            }
            else
            {
                // Mean squared error for other activation functions
                total_loss += (prediction - targets[i]).squaredNorm();
            }
        }
        return total_loss / inputs.size();
    } catch (const std::exception& e) {
        std::cerr << "Error in get_loss method: " << e.what() << std::endl;
        throw;
    }
}

// Apply regularization to the gradients
void NeuralNetwork::apply_regularization(std::vector<Eigen::MatrixXd>& weight_gradients,
                                         std::vector<Eigen::VectorXd>& bias_gradients)
{
    try {
        DEBUG_LOG("Applying regularization");
        switch (regularization_type) {
            case RegularizationType::L1:
                DEBUG_LOG("Applying L1 regularization");
                for (size_t i = 0; i < weights.size(); ++i) {
                    weight_gradients[i].array() += regularization_strength * weights[i].array().sign();
                }
                break;
            case RegularizationType::L2:
                DEBUG_LOG("Applying L2 regularization");
                for (size_t i = 0; i < weights.size(); ++i) {
                    weight_gradients[i].array() += regularization_strength * weights[i].array();
                }
                break;
            default:
                DEBUG_LOG("No regularization applied");
                break;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in apply_regularization method: " << e.what() << std::endl;
        throw;
    }
}

// Make a prediction for the given input
Eigen::VectorXd NeuralNetwork::predict(const Eigen::VectorXd &input) const
{
    return feedforward(input);
}

// Check if the matrix contains valid (finite) values
bool NeuralNetwork::is_valid(const Eigen::MatrixXd &mat) const
{
    return ((mat.array() == mat.array()).all() && (mat.array().abs() != std::numeric_limits<double>::infinity()).all());
}

// Check if the vector contains valid (finite) values
bool NeuralNetwork::is_valid(const Eigen::VectorXd &vec) const
{
    return ((vec.array() == vec.array()).all() && (vec.array().abs() != std::numeric_limits<double>::infinity()).all());
}

// Reset the neural network by reinitializing weights
void NeuralNetwork::reset()
{
    initialize_weights();
    // Reset any other necessary state
}

void NeuralNetwork::check_gradients(const Eigen::VectorXd &input, const Eigen::VectorXd &target)
{
    double epsilon = 1e-7;
    auto [weight_gradients, bias_gradients] = backpropagate(input, target);

    for (size_t l = 0; l < weights.size(); ++l)
    {
        for (int i = 0; i < weights[l].rows(); ++i)
        {
            for (int j = 0; j < weights[l].cols(); ++j)
            {
                double original_value = weights[l](i, j);

                weights[l](i, j) = original_value + epsilon;
                double loss_plus = get_loss({input}, {target});

                weights[l](i, j) = original_value - epsilon;
                double loss_minus = get_loss({input}, {target});

                weights[l](i, j) = original_value;

                double numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon);
                double backprop_gradient = weight_gradients[l](i, j);

                double relative_error = std::abs(numerical_gradient - backprop_gradient) /
                                        (std::abs(numerical_gradient) + std::abs(backprop_gradient) + 1e-15);

                if (relative_error > 1e-5)
                {
                    std::cout << "Gradient mismatch at layer " << l << ", weight (" << i << "," << j << ")" << std::endl;
                    std::cout << "Numerical: " << numerical_gradient << ", Backprop: " << backprop_gradient << std::endl;
                    std::cout << "Relative Error: " << relative_error << std::endl;
                }
            }
        }
    }
}