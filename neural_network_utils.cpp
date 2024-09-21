#include "neural_network.hpp"
#include "neural_network_common.hpp"

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
            d = std::normal_distribution<>(0.0, 0.05);
            break;
        case WeightInitialization::Xavier:
            d = std::normal_distribution<>(0.0, std::sqrt(1.0 / (fan_in + fan_out)));
            break;
        case WeightInitialization::He:
            d = std::normal_distribution<>(0.0, std::sqrt(1.0 / fan_in));
            break;
        }

        weights[i] = Eigen::MatrixXd::NullaryExpr(layers[i + 1], layers[i],
                                                  [&](){ return d(gen); });
        biases[i] = Eigen::VectorXd::NullaryExpr(layers[i + 1],
                                                 [&](){ return d(gen); });
    }

    batch_norms.clear();
    for (size_t i = 0; i < layers.size() - 1; ++i)
    {
        batch_norms.emplace_back(layers[i + 1]);
    }
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

bool NeuralNetwork::is_valid(const Eigen::MatrixXd &mat) const
{
    return ((mat.array() == mat.array()).all() && (mat.array().abs() != std::numeric_limits<double>::infinity()).all());
}

bool NeuralNetwork::is_valid(const Eigen::VectorXd &vec) const
{
    return ((vec.array() == vec.array()).all() && (vec.array().abs() != std::numeric_limits<double>::infinity()).all());
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