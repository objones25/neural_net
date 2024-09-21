#include "neural_network.hpp"
#include "neural_network_common.hpp"

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

        // Check if weights are initialized
        check_weights_initialization();

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
    } catch (const WeightInitializationError& e) {
        std::cerr << "Weight initialization error during training: " << e.what() << std::endl;
        throw;
    } catch (const SizeMismatchError& e) {
        std::cerr << "Size mismatch error during training: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error during training: " << e.what() << std::endl;
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
        std::vector<Eigen::MatrixXd> initial_weights = weights;
        std::vector<Eigen::VectorXd> z_values;
        std::vector<Eigen::VectorXd> bn_gamma_gradients(batch_norms.size());
        std::vector<Eigen::VectorXd> bn_beta_gradients(batch_norms.size());

        // Initialize gradients
        for (size_t i = 0; i < weights.size(); ++i) {
            weight_gradients[i] = Eigen::MatrixXd::Zero(weights[i].rows(), weights[i].cols());
            bias_gradients[i] = Eigen::VectorXd::Zero(biases[i].size());
        }

        // Compute gradients for the batch
        for (size_t i = 0; i < batch_inputs.size(); ++i) {
            auto [activations, batch_z_values] = feedforward_with_intermediates(batch_inputs[i]);
            z_values = batch_z_values;  // Store the last computed z_values
            auto [sample_weight_gradients, sample_bias_gradients] = backpropagate(batch_inputs[i], batch_targets[i]);

            for (size_t j = 0; j < weights.size(); ++j) {
                weight_gradients[j] += sample_weight_gradients[j];
                bias_gradients[j] += sample_bias_gradients[j];
            }
        }

        // Average the gradients
        for (size_t i = 0; i < weights.size(); ++i) {
            weight_gradients[i] /= batch_inputs.size();
            bias_gradients[i] /= batch_inputs.size();
        }

        // Compute batch norm gradients
        for (size_t i = 0; i < batch_norms.size(); ++i) {
            bn_gamma_gradients[i] = Eigen::VectorXd::Zero(layers[i + 1]);
            bn_beta_gradients[i] = Eigen::VectorXd::Zero(layers[i + 1]);
            for (const auto& input : batch_inputs) {
                auto [activations, batch_z_values] = feedforward_with_intermediates(input);
                bn_gamma_gradients[i] += (batch_z_values[i].array() * activations[i + 1].array()).matrix();
                bn_beta_gradients[i] += activations[i + 1];
            }
            bn_gamma_gradients[i] /= batch_inputs.size();
            bn_beta_gradients[i] /= batch_inputs.size();
        }

        // Apply regularization
        apply_regularization(weight_gradients, bias_gradients);

        // Update weights and biases
        for (size_t i = 0; i < weights.size(); ++i) {
            optimizer->update(weights[i], biases[i], weight_gradients[i], bias_gradients[i]);

            DEBUG_LOG("Weight update norm: " << (weights[i] - initial_weights[i]).norm());

            if (i < batch_norms.size()) {
                // Update batch normalization parameters
                Eigen::VectorXd mean = z_values[i].rowwise().mean();
                Eigen::VectorXd var = ((z_values[i].colwise() - mean).array().square().rowwise().sum() / z_values[i].cols()).sqrt();
                
                batch_norms[i].update_running_stats(mean, var);

                // Apply the gradients to gamma and beta
                Eigen::VectorXd gamma = batch_norms[i].get_gamma();
                Eigen::VectorXd beta = batch_norms[i].get_beta();
                gamma -= optimizer->get_learning_rate() * bn_gamma_gradients[i];
                beta -= optimizer->get_learning_rate() * bn_beta_gradients[i];
                batch_norms[i].set_gamma(gamma);
                batch_norms[i].set_beta(beta);

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

void NeuralNetwork::apply_regularization(std::vector<Eigen::MatrixXd>& weight_gradients,
                                         std::vector<Eigen::VectorXd>& bias_gradients)
{
    try {
        // Check if weights are initialized
        check_weights_initialization();

        // Check if sizes match
        if (weights.size() != weight_gradients.size()) {
            throw SizeMismatchError("Number of weight matrices (" + std::to_string(weights.size()) + 
                                    ") does not match number of gradient matrices (" + 
                                    std::to_string(weight_gradients.size()) + ")");
        }

        switch (regularization_type)
        {
        case RegularizationType::L1:
            for (size_t i = 0; i < weights.size(); ++i)
            {
                if (weights[i].rows() != weight_gradients[i].rows() || weights[i].cols() != weight_gradients[i].cols()) {
                    throw SizeMismatchError("Weight matrix size (" + std::to_string(weights[i].rows()) + "x" + 
                                            std::to_string(weights[i].cols()) + ") does not match gradient matrix size (" + 
                                            std::to_string(weight_gradients[i].rows()) + "x" + 
                                            std::to_string(weight_gradients[i].cols()) + ") at index " + std::to_string(i));
                }
                weight_gradients[i].array() += regularization_strength * weights[i].array().sign();
            }
            break;
        case RegularizationType::L2:
            for (size_t i = 0; i < weights.size(); ++i)
            {
                if (weights[i].rows() != weight_gradients[i].rows() || weights[i].cols() != weight_gradients[i].cols()) {
                    throw SizeMismatchError("Weight matrix size (" + std::to_string(weights[i].rows()) + "x" + 
                                            std::to_string(weights[i].cols()) + ") does not match gradient matrix size (" + 
                                            std::to_string(weight_gradients[i].rows()) + "x" + 
                                            std::to_string(weight_gradients[i].cols()) + ") at index " + std::to_string(i));
                }
                weight_gradients[i].array() += regularization_strength * weights[i].array();
            }
            break;
        default:
            // No regularization
            break;
        }
    } catch (const WeightInitializationError& e) {
        std::cerr << "Weight initialization error in apply_regularization: " << e.what() << std::endl;
        throw;
    } catch (const SizeMismatchError& e) {
        std::cerr << "Size mismatch error in apply_regularization: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error in apply_regularization: " << e.what() << std::endl;
        throw;
    }
}

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