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
        Logger::log("Initializing Neural Network", LogLevel::INFO);
        if (layer_sizes.size() < 2)
        {
            throw NetworkConfigurationError("Network must have at least an input and output layer");
        }

        if (layer_sizes.empty())
        {
            throw NetworkConfigurationError("Layer sizes vector cannot be empty");
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
            Logger::log("Added layer: " + std::to_string(layer_sizes[i]) + " -> " + std::to_string(layer_sizes[i + 1]), LogLevel::DEBUG);
        }
        Logger::log("Neural Network initialization complete", LogLevel::INFO);
    }
    catch (const std::exception &e)
    {
        Logger::error("Error in NeuralNetwork constructor: " + std::string(e.what()));
        throw;
    }
}

void NeuralNetwork::train(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y,
                          int epochs, int batch_size, double learning_rate,
                          double validation_split, int patience, double min_delta)
{
    try
    {
        Logger::log("Starting training", LogLevel::INFO);
        Logger::log("Input X shape: (" + std::to_string(X.rows()) + ", " +
                        std::to_string(X.cols()) + ")",
                    LogLevel::DEBUG);
        Logger::log("Target y shape: (" + std::to_string(y.rows()) + ", " +
                        std::to_string(y.cols()) + ")",
                    LogLevel::DEBUG);
        Logger::log("Epochs: " + std::to_string(epochs) + ", Batch size: " + std::to_string(batch_size), LogLevel::DEBUG);
        Logger::log("Learning rate: " + std::to_string(learning_rate), LogLevel::DEBUG);
        Logger::log("Validation split: " + std::to_string(validation_split) + ", Patience: " + std::to_string(patience), LogLevel::DEBUG);
        Logger::log("Min delta: " + std::to_string(min_delta), LogLevel::DEBUG);

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

        // Update learning rate for all layers
        this->set_learning_rate(learning_rate);

        int n_samples = X.rows();
        int n_validation = static_cast<int>(n_samples * validation_split);
        int n_train = n_samples - n_validation;

        Logger::log("Total samples: " + std::to_string(n_samples) +
                        ", Training samples: " + std::to_string(n_train) +
                        ", Validation samples: " + std::to_string(n_validation),
                    LogLevel::INFO);

        // Create shuffled indices
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        // Split data into training and validation sets
        Eigen::MatrixXd X_train(n_train, X.cols());
        Eigen::MatrixXd y_train(n_train, y.cols());
        Eigen::MatrixXd X_val(n_validation, X.cols());
        Eigen::MatrixXd y_val(n_validation, y.cols());

        for (int i = 0; i < n_train; ++i)
        {
            X_train.row(i) = X.row(indices[i]);
            y_train.row(i) = y.row(indices[i]);
        }
        for (int i = 0; i < n_validation; ++i)
        {
            X_val.row(i) = X.row(indices[n_train + i]);
            y_val.row(i) = y.row(indices[n_train + i]);
        }

        Logger::log("X_train shape: (" + std::to_string(X_train.rows()) + ", " +
                        std::to_string(X_train.cols()) + ")",
                    LogLevel::DEBUG);
        Logger::log("y_train shape: (" + std::to_string(y_train.rows()) + ", " +
                        std::to_string(y_train.cols()) + ")",
                    LogLevel::DEBUG);
        Logger::log("X_val shape: (" + std::to_string(X_val.rows()) + ", " +
                        std::to_string(X_val.cols()) + ")",
                    LogLevel::DEBUG);
        Logger::log("y_val shape: (" + std::to_string(y_val.rows()) + ", " +
                        std::to_string(y_val.cols()) + ")",
                    LogLevel::DEBUG);

        int n_batches = (n_train + batch_size - 1) / batch_size; // ceil division

        double best_val_loss = std::numeric_limits<double>::max();
        int epochs_no_improve = 0;

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            Logger::log("Starting epoch " + std::to_string(epoch + 1) + "/" + std::to_string(epochs), LogLevel::INFO);
            double epoch_loss = 0.0;

            // Shuffle the training data for each epoch
            std::vector<int> train_indices(n_train);
            std::iota(train_indices.begin(), train_indices.end(), 0);
            std::shuffle(train_indices.begin(), train_indices.end(), g);

            for (int batch = 0; batch < n_batches; ++batch)
            {
                int start_idx = batch * batch_size;
                int end_idx = std::min((batch + 1) * batch_size, n_train);
                int current_batch_size = end_idx - start_idx;

                Logger::log("Processing batch " + std::to_string(batch + 1) + "/" + std::to_string(n_batches), LogLevel::DEBUG);
                Logger::log("Current batch size: " + std::to_string(current_batch_size), LogLevel::DEBUG);

                Eigen::MatrixXd X_batch(current_batch_size, X_train.cols());
                Eigen::MatrixXd y_batch(current_batch_size, y_train.cols());

                for (int i = start_idx; i < end_idx; ++i)
                {
                    X_batch.row(i - start_idx) = X_train.row(train_indices[i]);
                    y_batch.row(i - start_idx) = y_train.row(train_indices[i]);
                }

                Logger::log("X_batch shape: (" + std::to_string(X_batch.rows()) + ", " +
                                std::to_string(X_batch.cols()) + ")",
                            LogLevel::DEBUG);
                Logger::log("y_batch shape: (" + std::to_string(y_batch.rows()) + ", " +
                                std::to_string(y_batch.cols()) + ")",
                            LogLevel::DEBUG);

                // Forward pass
                layers.front()->feedforward(X_batch);
                Eigen::MatrixXd output = layers.back()->get_last_output();

                Logger::log("Forward pass complete. Output shape: (" +
                                std::to_string(output.rows()) + ", " +
                                std::to_string(output.cols()) + ")",
                            LogLevel::DEBUG);

                // Calculate output gradient based on the loss function
                Eigen::MatrixXd output_gradient;
                switch (loss_function)
                {
                case LossFunction::MeanSquaredError:
                    output_gradient = output - y_batch;
                    break;
                case LossFunction::CrossEntropy:
                    output_gradient = -y_batch.array() / output.array() + (1 - y_batch.array()) / (1 - output.array());
                    break;
                default:
                    throw std::runtime_error("Unknown loss function");
                }

                Logger::log("Output gradient shape: (" +
                                std::to_string(output_gradient.rows()) + ", " +
                                std::to_string(output_gradient.cols()) + ")",
                            LogLevel::DEBUG);

                // Backward pass
                layers.back()->backpropagate(output_gradient, learning_rate);

                // Calculate loss
                double batch_loss = calculate_loss(X_batch, y_batch);
                epoch_loss += batch_loss;

                Logger::log("Batch " + std::to_string(batch + 1) + "/" + std::to_string(n_batches) +
                                " loss: " + std::to_string(batch_loss),
                            LogLevel::DEBUG);
            }

            epoch_loss /= n_batches;
            Logger::log("Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(epochs) +
                            ", Training Loss: " + std::to_string(epoch_loss),
                        LogLevel::INFO);

            // Calculate validation loss
            double val_loss = calculate_loss(X_val, y_val);
            Logger::log("Validation Loss: " + std::to_string(val_loss), LogLevel::INFO);

            // Early stopping check
            if (val_loss < best_val_loss - min_delta)
            {
                best_val_loss = val_loss;
                epochs_no_improve = 0;
                Logger::log("Validation loss improved. New best: " + std::to_string(best_val_loss), LogLevel::INFO);
            }
            else
            {
                epochs_no_improve++;
                Logger::log("Validation loss did not improve. Epochs without improvement: " +
                                std::to_string(epochs_no_improve),
                            LogLevel::INFO);

                if (epochs_no_improve >= patience)
                {
                    Logger::log("Early stopping triggered. Stopping training.", LogLevel::INFO);
                    break;
                }
            }
        }
        Logger::log("Training complete", LogLevel::INFO);
    }
    catch (const std::exception &e)
    {
        Logger::error("Error in NeuralNetwork::train: " + std::string(e.what()));
        throw;
    }
}

double NeuralNetwork::calculate_loss(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y)
{
    try
    {
        Logger::log("Calculating loss", LogLevel::DEBUG);
        Logger::log("Input X shape: (" + std::to_string(X.rows()) + ", " +
                        std::to_string(X.cols()) + ")",
                    LogLevel::DEBUG);
        Logger::log("Target y shape: (" + std::to_string(y.rows()) + ", " +
                        std::to_string(y.cols()) + ")",
                    LogLevel::DEBUG);

        if (X.rows() != y.rows())
        {
            throw TrainingDataError("Number of samples in X and y must match");
        }

        Logger::log("Starting feedforward pass", LogLevel::DEBUG);
        layers.front()->feedforward(X);
        Logger::log("Feedforward pass complete", LogLevel::DEBUG);

        Eigen::MatrixXd output = layers.back()->get_last_output();

        Logger::log("Network output shape: (" + std::to_string(output.rows()) + ", " +
                        std::to_string(output.cols()) + ")",
                    LogLevel::DEBUG);

        if (!output.allFinite())
        {
            throw NumericalInstabilityError("Non-finite values detected in network output");
        }

        double loss;
        switch (loss_function)
        {
        case LossFunction::MeanSquaredError:
        {
            Eigen::MatrixXd diff = output - y;
            loss = diff.array().square().sum() / (2 * X.rows());
        }
        break;
        case LossFunction::CrossEntropy:
        {
            const double epsilon = 1e-10; // Small value to avoid log(0)
            Eigen::ArrayXXd safe_output = output.array().max(epsilon).min(1 - epsilon);
            loss = -(y.array() * safe_output.log() + (1 - y.array()) * (1 - safe_output).log()).sum() / X.rows();
        }
        break;
        default:
            throw std::runtime_error("Unknown loss function");
        }

        if (!std::isfinite(loss))
        {
            throw NumericalInstabilityError("Non-finite loss value calculated");
        }

        Logger::log("Loss calculation complete: " + std::to_string(loss), LogLevel::DEBUG);
        return loss;
    }
    catch (const std::exception &e)
    {
        Logger::error("Error in NeuralNetwork::calculate_loss: " + std::string(e.what()));
        throw;
    }
}

void NeuralNetwork::set_learning_rate(double new_learning_rate)
{
    try
    {
        Logger::log("Setting new learning rate: " + std::to_string(new_learning_rate), LogLevel::INFO);
        learning_rate = new_learning_rate;
        for (auto &layer : layers)
        {
            layer->set_learning_rate(new_learning_rate);
        }
        Logger::log("Learning rate updated successfully", LogLevel::INFO);
    }
    catch (const std::exception &e)
    {
        Logger::error("Error in NeuralNetwork::set_learning_rate: " + std::string(e.what()));
        throw;
    }
}

double NeuralNetwork::get_learning_rate() const
{
    return learning_rate;
}