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