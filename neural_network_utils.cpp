#include "neural_network.hpp"
#include "neural_network_common.hpp"

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

    for (size_t l = 0; l < layers.size(); ++l)
    {
        for (int i = 0; i < layers[l].weights.rows(); ++i)
        {
            for (int j = 0; j < layers[l].weights.cols(); ++j)
            {
                double original_value = layers[l].weights(i, j);

                layers[l].weights(i, j) = original_value + epsilon;
                double loss_plus = get_loss({input}, {target});

                layers[l].weights(i, j) = original_value - epsilon;
                double loss_minus = get_loss({input}, {target});

                layers[l].weights(i, j) = original_value;

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

void NeuralNetwork::check_weights_initialization() const
{
    for (size_t i = 0; i < layers.size(); ++i)
    {
        if (layers[i].weights.size() == 0)
        {
            throw WeightInitializationError("Weights are not properly initialized at layer " + std::to_string(i));
        }

        if (!is_valid(layers[i].weights))
        {
            throw WeightInitializationError("Invalid weight values detected at layer " + std::to_string(i));
        }

        // Check if weights are all zero or very close to zero
        if (layers[i].weights.isZero(1e-10))
        {
            throw WeightInitializationError("Weights are all zero or very close to zero at layer " + std::to_string(i));
        }

        // Check for NaN or Inf values
        if (layers[i].weights.hasNaN() || !layers[i].weights.allFinite())
        {
            throw WeightInitializationError("NaN or Inf values detected in weights at layer " + std::to_string(i));
        }
    }
}

void NeuralNetwork::set_weights(const std::vector<Eigen::MatrixXd>& new_weights) {
    if (new_weights.size() != layers.size()) {
        throw std::invalid_argument("Invalid number of weight matrices");
    }
    for (size_t i = 0; i < layers.size(); ++i) {
        if (new_weights[i].rows() != layers[i].weights.rows() || new_weights[i].cols() != layers[i].weights.cols()) {
            throw std::invalid_argument("Invalid dimensions for weight matrix " + std::to_string(i));
        }
        layers[i].weights = new_weights[i];
    }
}

void NeuralNetwork::set_biases(const std::vector<Eigen::VectorXd>& new_biases) {
    if (new_biases.size() != layers.size()) {
        throw std::invalid_argument("Invalid number of bias vectors");
    }
    for (size_t i = 0; i < layers.size(); ++i) {
        if (new_biases[i].size() != layers[i].biases.size()) {
            throw std::invalid_argument("Invalid dimensions for bias vector " + std::to_string(i));
        }
        layers[i].biases = new_biases[i];
    }
}