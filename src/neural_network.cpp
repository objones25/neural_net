#include "neural_network.hpp"
#include "logger.hpp"
#include "exceptions.hpp"
#include <iostream>
#include <random>
#include <algorithm>

NeuralNetwork::NeuralNetwork(const std::vector<int> &layer_sizes,
                             ActivationType hidden_activation,
                             ActivationType output_activation,
                             const std::string &optimizer_name,
                             double learning_rate,
                             LossFunction loss_function,
                             bool use_batch_norm,
                             WeightInitialization weight_init)
    : learning_rate(learning_rate),
      loss_function(loss_function),
      hidden_activation(hidden_activation),
      output_activation(output_activation)
{
    try
    {
        Logger::log("Initializing Neural Network");
        if (layer_sizes.size() < 2)
        {
            throw NetworkConfigurationError("Network must have at least an input and output layer");
        }

        // Create and connect layers
        for (size_t i = 0; i < layer_sizes.size() - 1; ++i)
        {
            ActivationType activation = (i == layer_sizes.size() - 2) ? output_activation : hidden_activation;
            auto optimizer = create_optimizer(optimizer_name, learning_rate);
            auto layer = std::make_shared<Layer>(layer_sizes[i], layer_sizes[i + 1], activation,
                                                 std::move(optimizer), learning_rate,
                                                 use_batch_norm, 0.99, weight_init);

            if (!layers.empty())
            {
                layers.back()->set_next_layer(layer);
                layer->set_prev_layer(layers.back());
            }

            layers.push_back(layer);
            Logger::log("Added layer: " + std::to_string(layer_sizes[i]) + " -> " + std::to_string(layer_sizes[i + 1]));
        }
        Logger::log("Neural Network initialization complete");
    }
    catch (const std::exception &e)
    {
        Logger::log_exception(e, "NeuralNetwork constructor");
        throw;
    }
}

void NeuralNetwork::train(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y,
                          int epochs, int batch_size, double learning_rate)
{
    try
    {
        Logger::log("Starting training");
        if (X.rows() != y.rows())
        {
            throw TrainingDataError("Number of samples in X and y must match");
        }
        if (X.cols() != layers.front()->get_input_size())
        {
            throw SizeMismatchError("Input size does not match network input layer size");
        }
        if (y.cols() != layers.back()->get_output_size())
        {
            throw SizeMismatchError("Output size does not match network output layer size");
        }

        int n_samples = X.rows();
        int n_batches = (n_samples + batch_size - 1) / batch_size; // ceil division

        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            Logger::log("Starting epoch " + std::to_string(epoch + 1) + "/" + std::to_string(epochs));
            double epoch_loss = 0.0;

            // Shuffle the indices
            std::shuffle(indices.begin(), indices.end(), g);

            for (int batch = 0; batch < n_batches; ++batch)
            {
                int start_idx = batch * batch_size;
                int end_idx = std::min((batch + 1) * batch_size, n_samples);

                Eigen::MatrixXd X_batch(end_idx - start_idx, X.cols());
                Eigen::MatrixXd y_batch(end_idx - start_idx, y.cols());

                for (int i = start_idx; i < end_idx; ++i)
                {
                    X_batch.row(i - start_idx) = X.row(indices[i]);
                    y_batch.row(i - start_idx) = y.row(indices[i]);
                }

                // Forward pass
                double batch_loss = layers.front()->feedforward(X_batch);

                // Backward pass
                Eigen::MatrixXd output_gradient = layers.back()->get_last_output() - y_batch;
                layers.back()->backpropagate(output_gradient, learning_rate);

                epoch_loss += batch_loss;
            }

            epoch_loss /= n_batches;
            Logger::log("Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(epochs) + ", Loss: " + std::to_string(epoch_loss));

            // Optionally, implement early stopping here
        }
        Logger::log("Training complete");
    }
    catch (const std::exception &e)
    {
        Logger::log_exception(e, "NeuralNetwork::train");
        throw;
    }
}

double NeuralNetwork::calculate_loss(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y)
{
    try
    {
        Logger::log("Calculating loss");
        if (X.rows() != y.rows())
        {
            throw TrainingDataError("Number of samples in X and y must match");
        }

        layers.front()->feedforward(X);
        Eigen::MatrixXd output = layers.back()->get_last_output();

        double loss;
        switch (loss_function)
        {
        case LossFunction::MeanSquaredError:
            loss = (output - y).array().square().sum() / (2 * X.rows());
            break;
        case LossFunction::CrossEntropy:
            loss = -(y.array() * output.array().log()).sum() / X.rows();
            break;
        default:
            throw std::runtime_error("Unknown loss function");
        }
        Logger::log("Loss calculation complete: " + std::to_string(loss));
        return loss;
    }
    catch (const std::exception &e)
    {
        Logger::log_exception(e, "NeuralNetwork::calculate_loss");
        throw;
    }
}

void NeuralNetwork::set_learning_rate(double new_learning_rate)
{
    try
    {
        Logger::log("Setting new learning rate: " + std::to_string(new_learning_rate));
        learning_rate = new_learning_rate;
        for (auto &layer : layers)
        {
            layer->set_learning_rate(new_learning_rate);
        }
        Logger::log("Learning rate updated successfully");
    }
    catch (const std::exception &e)
    {
        Logger::log_exception(e, "NeuralNetwork::set_learning_rate");
        throw;
    }
}

double NeuralNetwork::get_learning_rate() const
{
    return learning_rate;
}