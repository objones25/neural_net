#include "neural_network.hpp"
#include "neural_network_common.hpp"

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes,
                             ActivationFunction::Type hidden_activation,
                             ActivationFunction::Type output_activation,
                             WeightInitialization weight_init,
                             const std::string& optimizer_name,
                             double learning_rate,
                             RegularizationType reg_type,
                             double reg_strength,
                             double learning_rate_adjustment)
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

    // Initialize batch_norms vector (excluding the output layer)
    batch_norms.reserve(layers.size() - 2);
    for (size_t i = 1; i < layers.size() - 1; ++i)
    {
        batch_norms.emplace_back(layers[i]);
    }

    // Initialize weights and biases vectors
    weights.resize(layers.size() - 1);
    biases.resize(layers.size() - 1);

    std::cout << "Creating optimizer" << std::endl;
    // Use a slightly larger learning rate
    double adjusted_learning_rate = learning_rate * learning_rate_adjustment;
    optimizer = create_optimizer_for_network(optimizer_name, adjusted_learning_rate);
    std::cout << "Optimizer created with adjusted learning rate: " << adjusted_learning_rate << std::endl;

    // Initialize weights and biases
    initialize_weights();
    std::cout << "Weights initialized" << std::endl;

    std::cout << "NeuralNetwork constructor completed" << std::endl;
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