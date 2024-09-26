#include "neural_network.hpp"
#include "neural_network_common.hpp"

void NeuralNetwork::train(const std::vector<Eigen::VectorXd> &inputs,
                          const std::vector<Eigen::VectorXd> &targets,
                          int epochs,
                          int batch_size,
                          double error_tolerance)
{
    try
    {
        debug_print("Starting training");
        if (inputs.empty() || targets.empty())
        {
            throw std::invalid_argument("Inputs and targets cannot be empty");
        }
        if (inputs.size() != targets.size())
        {
            throw std::invalid_argument("Number of inputs must match number of targets");
        }
        if (epochs <= 0 || batch_size <= 0)
        {
            throw std::invalid_argument("Invalid training parameters");
        }

        // Check if weights are initialized
        check_weights_initialization();

        debug_print("Starting training with " + std::to_string(epochs) + " epochs and batch size " + std::to_string(batch_size));

        std::vector<size_t> indices(inputs.size());
        std::iota(indices.begin(), indices.end(), 0);

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            if (lr_scheduler)
            {
                double new_lr = lr_scheduler(epoch);
                optimizer->setLearningRate(new_lr);
            }
            debug_print("Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(epochs));

            for (size_t i = 0; i < inputs.size(); i += batch_size)
            {
                size_t batch_end = std::min(i + batch_size, inputs.size());
                std::vector<Eigen::VectorXd> batch_inputs, batch_targets;

                for (size_t j = i; j < batch_end; ++j)
                {
                    batch_inputs.push_back(inputs[indices[j]]);
                    batch_targets.push_back(targets[indices[j]]);
                }

                try
                {
                    update_batch(batch_inputs, batch_targets);

                    // Perform gradient checking periodically (e.g., every 100 batches)
                    if (i % (100 * batch_size) == 0)
                    {
                        check_gradients(batch_inputs[0], batch_targets[0]);
                    }
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error in update_batch: " << e.what() << std::endl;
                    throw;
                }
            }

            double loss;
            try
            {
                loss = get_loss(inputs, targets);
                std::cout << "Epoch " << epoch + 1 << "/" << epochs << " Loss: " << loss << std::endl;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error calculating loss: " << e.what() << std::endl;
                throw;
            }

            if (loss < error_tolerance)
            {
                debug_print("Reached error tolerance. Stopping training.");
                break;
            }
        }
        debug_print("Training completed");
    }
    catch (const std::exception &e)
    {
        std::cerr << "Unexpected error during training: " << e.what() << std::endl;
        throw;
    }
}

void NeuralNetwork::update_batch(const std::vector<Eigen::VectorXd> &batch_inputs,
                                 const std::vector<Eigen::VectorXd> &batch_targets)
{
    if (batch_inputs.empty() || batch_targets.empty())
    {
        throw std::invalid_argument("Batch inputs and targets cannot be empty");
    }
    if (batch_inputs.size() != batch_targets.size())
    {
        throw std::invalid_argument("Number of inputs must match number of targets in batch");
    }

    const size_t batch_size = batch_inputs.size();
    std::vector<Eigen::MatrixXd> weight_gradients(layers.size());
    std::vector<Eigen::VectorXd> bias_gradients(layers.size());

    // Initialize gradients
    for (size_t i = 0; i < layers.size(); ++i)
    {
        weight_gradients[i] = Eigen::MatrixXd::Zero(layers[i].weights.rows(), layers[i].weights.cols());
        bias_gradients[i] = Eigen::VectorXd::Zero(layers[i].biases.size());
    }

    // Compute gradients for the batch
    for (size_t i = 0; i < batch_size; ++i)
    {
        auto [sample_weight_gradients, sample_bias_gradients] = backpropagate(batch_inputs[i], batch_targets[i]);

        for (size_t j = 0; j < layers.size(); ++j)
        {
            weight_gradients[j] += sample_weight_gradients[j];
            bias_gradients[j] += sample_bias_gradients[j];
        }
    }

    // Average the gradients
    for (size_t i = 0; i < layers.size(); ++i)
    {
        weight_gradients[i] /= static_cast<double>(batch_size);
        bias_gradients[i] /= static_cast<double>(batch_size);
    }

    // Apply regularization
    apply_regularization(weight_gradients, bias_gradients);

    // Clip gradients
    clip_gradients(weight_gradients, bias_gradients);

    // Update weights and biases
    for (size_t i = 0; i < layers.size(); ++i)
    {
        try
        {
            optimizer->update(layers[i], weight_gradients[i], bias_gradients[i]);
        }
        catch (const OptimizerError &e)
        {
            throw OptimizerError("Error updating layer " + std::to_string(i) + ": " + e.what());
        }
    }
}

void NeuralNetwork::apply_regularization(std::vector<Eigen::MatrixXd> &weight_gradients,
                                         std::vector<Eigen::VectorXd> &bias_gradients)
{
    if (layers.size() != weight_gradients.size()) {
        throw SizeMismatchError("Number of layers does not match number of gradient matrices");
    }

    switch (regularization_type)
    {
    case RegularizationType::L1:
        #pragma omp parallel for
        for (size_t i = 0; i < layers.size(); ++i)
        {
            if (layers[i].weights.rows() != weight_gradients[i].rows() || layers[i].weights.cols() != weight_gradients[i].cols()) {
                throw SizeMismatchError("Weight matrix size mismatch at index " + std::to_string(i));
            }
            weight_gradients[i] += regularization_strength * layers[i].weights.unaryExpr([](double x) {
                return x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0);  // Handle zero case
            });
        }
        break;
    case RegularizationType::L2:
        #pragma omp parallel for
        for (size_t i = 0; i < layers.size(); ++i)
        {
            if (layers[i].weights.rows() != weight_gradients[i].rows() || layers[i].weights.cols() != weight_gradients[i].cols()) {
                throw SizeMismatchError("Weight matrix size mismatch at index " + std::to_string(i));
            }
            weight_gradients[i] += regularization_strength * layers[i].weights;
        }
        break;
    default:
        // No regularization
        break;
    }

    // Clip gradients after applying regularization
    for (size_t i = 0; i < layers.size(); ++i)
    {
        clip_and_check(weight_gradients[i], "Weight gradients after regularization", 1e3);
        clip_and_check(bias_gradients[i], "Bias gradients after regularization", 1e3);
    }
}

double NeuralNetwork::get_loss(const std::vector<Eigen::VectorXd> &inputs,
                               const std::vector<Eigen::VectorXd> &targets) const
{
    try
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
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error in get_loss method: " << e.what() << std::endl;
        throw;
    }
}

void NeuralNetwork::clip_gradients(std::vector<Eigen::MatrixXd>& weight_gradients,
                                   std::vector<Eigen::VectorXd>& bias_gradients,
                                   double max_norm) {
    double total_norm = 0.0;
    for (size_t i = 0; i < weight_gradients.size(); ++i) {
        total_norm += weight_gradients[i].norm();
        total_norm += bias_gradients[i].norm();
    }

    if (total_norm > max_norm) {
        double scale = max_norm / total_norm;
        for (size_t i = 0; i < weight_gradients.size(); ++i) {
            weight_gradients[i] *= scale;
            bias_gradients[i] *= scale;
        }
    }
}

void NeuralNetwork::set_learning_rate_scheduler(LearningRateScheduler scheduler)
{
    lr_scheduler = scheduler;
}