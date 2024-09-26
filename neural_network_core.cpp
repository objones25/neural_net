#include "neural_network.hpp"
#include "neural_network_common.hpp"

NeuralNetwork::NeuralNetwork(const std::vector<int> &layer_sizes,
                             ActivationFunction::Type hidden_activation,
                             ActivationFunction::Type output_activation,
                             WeightInitialization weight_init,
                             const std::string &optimizer_name,
                             double learning_rate,
                             RegularizationType reg_type,
                             double reg_strength,
                             double learning_rate_adjustment,
                             bool use_batch_norm)
    : layer_sizes(layer_sizes),
      activation_function(hidden_activation, output_activation),
      weight_init(weight_init),
      regularization_type(reg_type),
      regularization_strength(reg_strength),
      use_batch_norm(use_batch_norm)
{
    debug_print("NeuralNetwork constructor started");
    debug_print("Hidden activation type: " + std::to_string(static_cast<int>(hidden_activation)));
    debug_print("Output activation type: " + std::to_string(static_cast<int>(output_activation)));

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

    // Print layer sizes for debugging
    std::cout << "Layer sizes: ";
    for (size_t i = 0; i < layer_sizes.size(); ++i)
    {
        std::cout << layer_sizes[i] << " ";
    }
    std::cout << std::endl;

    // Initialize layers
    layers.reserve(layer_sizes.size() - 1);
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i)
    {
        layers.emplace_back(layer_sizes[i], layer_sizes[i + 1], use_batch_norm && i < layer_sizes.size() - 2);
    }

    debug_print("Creating optimizer");
    double adjusted_learning_rate = learning_rate * learning_rate_adjustment;
    optimizer = create_optimizer_for_network(optimizer_name, adjusted_learning_rate);
    debug_print("Optimizer created with adjusted learning rate: " + std::to_string(adjusted_learning_rate));

    // Initialize weights and biases
    initialize_weights();
    debug_print("Weights initialized");

    debug_print("NeuralNetwork constructor completed");
}

// Static method implementation
std::unique_ptr<OptimizationAlgorithm> NeuralNetwork::create_optimizer_for_network(const std::string &name, double learning_rate)
{
    return create_optimizer(name, learning_rate);
}

void NeuralNetwork::reset()
{
    initialize_weights();
    // Reset any other necessary state
}

void NeuralNetwork::initialize_weights()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    for (auto &layer : layers)
    {
        int fan_in = layer.weights.cols();
        int fan_out = layer.weights.rows();

        std::normal_distribution<> d;
        switch (weight_init)
        {
        case WeightInitialization::Random:
            d = std::normal_distribution<>(0.0, 0.05);
            break;
        case WeightInitialization::Xavier:
            d = std::normal_distribution<>(0.0, std::sqrt(2.0 / (fan_in + fan_out)));
            break;
        case WeightInitialization::He:
            d = std::normal_distribution<>(0.0, std::sqrt(2.0 / fan_in));
            break;
        }

        layer.weights = Eigen::MatrixXd::NullaryExpr(fan_out, fan_in,
                                                     [&]()
                                                     { return d(gen); });
        layer.biases = Eigen::VectorXd::NullaryExpr(fan_out,
                                                    [&]()
                                                    { return d(gen); });

        if (!is_valid(layer.weights) || !is_valid(layer.biases))
        {
            throw WeightInitializationError("Invalid values detected during weight initialization");
        }
    }
}

void NeuralNetwork::validate() const
{
    if (layer_sizes.size() < 2)
    {
        throw NetworkConfigurationError("Network must have at least two layers");
    }
    if (layers.size() != layer_sizes.size() - 1)
    {
        throw NetworkConfigurationError("Number of layers must match number of layer sizes minus one");
    }

    for (size_t i = 0; i < layers.size(); ++i)
    {
        if (layers[i].weights.rows() != layer_sizes[i + 1] || layers[i].weights.cols() != layer_sizes[i])
        {
            throw NetworkConfigurationError("Weight matrix dimensions mismatch");
        }
        if (layers[i].biases.size() != layer_sizes[i + 1])
        {
            throw NetworkConfigurationError("Bias vector size mismatch");
        }
    }
}