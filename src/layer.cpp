#include "layer.hpp"

Layer::Layer(int input_size, int output_size, ActivationType activation,
             std::unique_ptr<Optimizer> optimizer,
             double learning_rate,
             bool use_batch_norm, double momentum, WeightInitialization weight_init)
    : input_size(input_size),
      output_size(output_size),
      momentum(momentum),
      use_batch_norm(use_batch_norm),
      activation_type(activation),
      optimizer(std::move(optimizer)),
      learning_rate(learning_rate)
{
    try
    {
        Logger::log("Initializing Layer", LogLevel::INFO);
        if (input_size <= 0 || output_size <= 0)
        {
            throw std::invalid_argument("Input and output sizes must be positive");
        }
        if (learning_rate <= 0)
        {
            throw std::invalid_argument("Learning rate must be positive");
        }
        if (momentum < 0 || momentum >= 1)
        {
            throw std::invalid_argument("Momentum must be in the range [0, 1)");
        }

        // Initialize weights based on the specified method
        std::random_device rd;
        std::mt19937 gen(rd());

        switch (weight_init)
        {
        case WeightInitialization::Xavier:
        {
            std::normal_distribution<> d(0, std::sqrt(2.0 / (input_size + output_size)));
            weights = Eigen::MatrixXd::NullaryExpr(output_size, input_size, [&]()
                                                   { return d(gen); });
            break;
        }
        case WeightInitialization::He:
        {
            std::normal_distribution<> d(0, std::sqrt(2.0 / input_size));
            weights = Eigen::MatrixXd::NullaryExpr(output_size, input_size, [&]()
                                                   { return d(gen); });
            break;
        }
        case WeightInitialization::LeCun:
        {
            std::normal_distribution<> d(0, std::sqrt(1.0 / input_size));
            weights = Eigen::MatrixXd::NullaryExpr(output_size, input_size, [&]()
                                                   { return d(gen); });
            break;
        }
        default:
        {
            // Default to Xavier initialization
            std::normal_distribution<> d(0, std::sqrt(2.0 / (input_size + output_size)));
            weights = Eigen::MatrixXd::NullaryExpr(output_size, input_size, [&]()
                                                   { return d(gen); });
        }
        }

        biases = Eigen::VectorXd::Zero(output_size);

        if (use_batch_norm)
        {
            gamma = Eigen::VectorXd::Ones(output_size);
            beta = Eigen::VectorXd::Zero(output_size);
            running_mean = Eigen::VectorXd::Zero(output_size);
            running_variance = Eigen::VectorXd::Ones(output_size);
        }

        this->optimizer->setLearningRate(learning_rate);

        if (!weights.allFinite() || !biases.allFinite())
        {
            Logger::error("Non-finite values detected in initial weights or biases");
            throw NumericalInstabilityError("Non-finite values detected in initial weights or biases");
        }
    }
    catch (const std::exception &e)
    {
        Logger::error("Error in Layer constructor: " + std::string(e.what()));
        throw;
    }
}

void Layer::set_prev_layer(std::shared_ptr<Layer> layer)
{
    prev_layer = layer;
}

void Layer::set_next_layer(std::shared_ptr<Layer> layer)
{
    next_layer = layer;
}

void Layer::clip_gradients(Eigen::MatrixXd &d_weights, Eigen::VectorXd &d_biases,
                           Eigen::VectorXd *d_gamma, Eigen::VectorXd *d_beta,
                           double clip_value)
{
    // Clip gradients for weights
    double weights_norm = d_weights.norm();
    if (weights_norm > clip_value)
    {
        d_weights *= (clip_value / weights_norm);
    }

    // Clip gradients for biases
    double biases_norm = d_biases.norm();
    if (biases_norm > clip_value)
    {
        d_biases *= (clip_value / biases_norm);
    }

    // If using batch normalization, clip gradients for gamma and beta
    if (d_gamma && d_beta)
    {
        double gamma_norm = d_gamma->norm();
        if (gamma_norm > clip_value)
        {
            *d_gamma *= (clip_value / gamma_norm);
        }

        double beta_norm = d_beta->norm();
        if (beta_norm > clip_value)
        {
            *d_beta *= (clip_value / beta_norm);
        }
    }
}

double Layer::feedforward(const Eigen::MatrixXd &input)
{
    try
    {
        const double epsilon = 1e-8;

        Logger::log("Feedforward called with input shape: (" +
                    std::to_string(input.rows()) + ", " +
                    std::to_string(input.cols()) + ")",
                    LogLevel::DEBUG);
        Logger::log("Layer input size: " + std::to_string(input_size) +
                    ", output size: " + std::to_string(output_size),
                    LogLevel::DEBUG);
        Logger::log("Weights shape: (" + std::to_string(weights.rows()) + ", " +
                    std::to_string(weights.cols()) + ")",
                    LogLevel::DEBUG);
        Logger::log("Biases shape: (" + std::to_string(biases.size()) + ")", LogLevel::DEBUG);

        if (input.cols() != input_size)
        {
            throw SizeMismatchError("Input size mismatch. Expected: " +
                                    std::to_string(input_size) +
                                    ", Got: " + std::to_string(input.cols()));
        }

        last_input = input;
        Eigen::MatrixXd z = input * weights.transpose();
        Logger::log("After matrix multiplication, z shape: (" +
                    std::to_string(z.rows()) + ", " +
                    std::to_string(z.cols()) + ")",
                    LogLevel::DEBUG);

        z.rowwise() += biases.transpose();
        Logger::log("After adding biases, z shape: (" +
                    std::to_string(z.rows()) + ", " +
                    std::to_string(z.cols()) + ")",
                    LogLevel::DEBUG);

        if (use_batch_norm)
        {
            if (input.rows() == 1)
            {
                Logger::log("Single sample batch normalization", LogLevel::DEBUG);
                z = ((z.array().rowwise() - running_mean.transpose().array()) /
                     (running_variance.array() + epsilon).sqrt().transpose())
                        .matrix();
                z = (z.array().rowwise() * gamma.transpose().array()).matrix();
                z.rowwise() += beta.transpose();
            }
            else
            {
                Logger::log("Batch normalization for " + std::to_string(input.rows()) + " samples", LogLevel::DEBUG);
                Eigen::VectorXd batch_mean = z.colwise().mean();
                Eigen::VectorXd batch_var = ((z.rowwise() - batch_mean.transpose()).array().square().colwise().sum() /
                                             (z.rows() - 1))
                                                .matrix();

                running_mean = momentum * running_mean + (1 - momentum) * batch_mean;
                running_variance = momentum * running_variance + (1 - momentum) * batch_var;

                normalized_input = (z.rowwise() - batch_mean.transpose()).array().rowwise() /
                                   (batch_var.array() + epsilon).sqrt().transpose();

                z = (normalized_input.array().rowwise() * gamma.transpose().array()).matrix();
                z.rowwise() += beta.transpose();
            }
            Logger::log("After batch norm, z shape: (" +
                        std::to_string(z.rows()) + ", " +
                        std::to_string(z.cols()) + ")",
                        LogLevel::DEBUG);
            Logger::log("Gamma shape: (" + std::to_string(gamma.size()) + ")", LogLevel::DEBUG);
            Logger::log("Beta shape: (" + std::to_string(beta.size()) + ")", LogLevel::DEBUG);
            Logger::log("Running mean shape: (" + std::to_string(running_mean.size()) + ")", LogLevel::DEBUG);
            Logger::log("Running variance shape: (" + std::to_string(running_variance.size()) + ")", LogLevel::DEBUG);
        }

        Logger::log("Before activation, z shape: (" +
                    std::to_string(z.rows()) + ", " +
                    std::to_string(z.cols()) + ")",
                    LogLevel::DEBUG);
        last_output = activate(z);
        Logger::log("After activation, last_output shape: (" +
                    std::to_string(last_output.rows()) + ", " +
                    std::to_string(last_output.cols()) + ")",
                    LogLevel::DEBUG);

        if (next_layer)
        {
            Logger::log("Passing to next layer", LogLevel::DEBUG);
            return next_layer->feedforward(last_output) + compute_regularization_loss();
        }
        
        Logger::log("Finished feedforward in Layer", LogLevel::DEBUG);
        return compute_regularization_loss();
    }
    catch (const std::exception &e)
    {
        Logger::error("Error in Layer::feedforward: " + std::string(e.what()));
        throw;
    }
}

void Layer::backpropagate(const Eigen::MatrixXd &output_gradient, double learning_rate)
{
    try
    {
        const double epsilon = 1e-8;

        Logger::log("Starting backpropagation in Layer", LogLevel::DEBUG);
        Logger::log("Backprop input gradient shape: (" +
                    std::to_string(output_gradient.rows()) + ", " +
                    std::to_string(output_gradient.cols()) + ")",
                    LogLevel::DEBUG);
        Logger::log("Weights shape: (" +
                    std::to_string(weights.rows()) + ", " +
                    std::to_string(weights.cols()) + ")",
                    LogLevel::DEBUG);
        Logger::log("Last input shape: (" +
                    std::to_string(last_input.rows()) + ", " +
                    std::to_string(last_input.cols()) + ")",
                    LogLevel::DEBUG);
        Logger::log("Last output shape: (" +
                    std::to_string(last_output.rows()) + ", " +
                    std::to_string(last_output.cols()) + ")",
                    LogLevel::DEBUG);

        if (output_gradient.cols() != output_size)
        {
            throw SizeMismatchError("Output gradient size does not match layer output size in backpropagate");
        }

        int batch_size = last_input.rows();
        Logger::log("Batch size: " + std::to_string(batch_size), LogLevel::DEBUG);

        // Compute the gradient of the activation function
        Eigen::MatrixXd d_activation = activate_derivative(last_output);
        Logger::log("Activation derivative shape: (" +
                    std::to_string(d_activation.rows()) + ", " +
                    std::to_string(d_activation.cols()) + ")",
                    LogLevel::DEBUG);
        
        // Element-wise multiplication of output gradient and activation gradient
        Eigen::MatrixXd d_output = output_gradient.array() * d_activation.array();
        
        Logger::log("d_output shape: (" +
                    std::to_string(d_output.rows()) + ", " +
                    std::to_string(d_output.cols()) + ")",
                    LogLevel::DEBUG);

        Eigen::MatrixXd d_input;
        Eigen::VectorXd d_gamma, d_beta;

        if (use_batch_norm)
        {
            if (batch_size == 1)
            {
                Logger::log("Warning: Batch size of 1, using running statistics for batch normalization", LogLevel::WARNING);
                d_input = d_output.array().rowwise() * (gamma.array() / (running_variance.array() + epsilon).sqrt()).transpose();
            }
            else
            {
                d_gamma = (normalized_input.array() * d_output.array()).colwise().sum();
                d_beta = d_output.colwise().sum();

                Eigen::MatrixXd d_normalized = d_output.array().rowwise() * gamma.transpose().array();

                Eigen::VectorXd batch_var = ((last_input.rowwise() - running_mean.transpose()).array().square().colwise().sum() / batch_size).matrix();
                Eigen::VectorXd d_var = (d_normalized.array() * (last_input.rowwise() - running_mean.transpose()).array()).colwise().sum() *
                                        (-0.5 * (batch_var.array() + epsilon).pow(-1.5));

                Eigen::VectorXd d_mean = (d_normalized.array() * -1 / (batch_var.array() + epsilon).sqrt()).colwise().sum().matrix() +
                                         (d_var.array() * -2 * (last_input.rowwise() - running_mean.transpose()).colwise().sum().array() / batch_size).matrix();

                d_input = (d_normalized.array() / (batch_var.array() + epsilon).sqrt().transpose().replicate(batch_size, 1) +
                           d_var.transpose().replicate(batch_size, 1).array() * 2 * (last_input.rowwise() - running_mean.transpose()).array() / batch_size +
                           d_mean.transpose().replicate(batch_size, 1).array() / batch_size)
                              .matrix();
            }
        }
        else
        {
            d_input = d_output;
        }

        Logger::log("d_input shape: (" +
                    std::to_string(d_input.rows()) + ", " +
                    std::to_string(d_input.cols()) + ")",
                    LogLevel::DEBUG);

        if (!d_input.allFinite())
        {
            throw NumericalInstabilityError("Non-finite values detected in input gradients");
        }

        Eigen::MatrixXd d_weights = last_input.transpose() * d_input / batch_size;
        Eigen::VectorXd d_biases = d_input.colwise().mean();

        Logger::log("d_weights shape: (" + std::to_string(d_weights.rows()) + ", " +
                    std::to_string(d_weights.cols()) + ")",
                    LogLevel::DEBUG);
        Logger::log("d_biases shape: (" + std::to_string(d_biases.rows()) + ", " +
                    std::to_string(d_biases.cols()) + ")",
                    LogLevel::DEBUG);

        if (!d_weights.allFinite() || !d_biases.allFinite())
        {
            throw NumericalInstabilityError("Non-finite values detected in weight or bias gradients");
        }

        // Add regularization gradient
        d_weights += compute_regularization_gradient();

        try
        {
            if (use_batch_norm)
            {
                clip_gradients(d_weights, d_biases, &d_gamma, &d_beta);
            }
            else
            {
                clip_gradients(d_weights, d_biases);
            }
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Error during gradient clipping: " + std::string(e.what()));
        }

        optimizer->update(weights, biases, d_weights, d_biases);

        if (use_batch_norm)
        {
            gamma -= learning_rate * d_gamma / batch_size;
            beta -= learning_rate * d_beta / batch_size;
        }

        if (!weights.allFinite() || !biases.allFinite())
        {
            throw NumericalInstabilityError("Non-finite values detected in weights or biases after update");
        }

        Eigen::MatrixXd prev_layer_gradient = d_input * weights;

        if (!prev_layer_gradient.allFinite())
        {
            throw NumericalInstabilityError("Non-finite values detected in previous layer gradient");
        }

        if (prev_layer)
        {
            try
            {
                prev_layer->backpropagate(prev_layer_gradient, learning_rate);
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error("Error in previous layer during backpropagation: " + std::string(e.what()));
            }
        }
        Logger::log("Finished backpropagation in Layer", LogLevel::DEBUG);
    }
    catch (const std::exception &e)
    {
        Logger::error("Error in Layer::backpropagate: " + std::string(e.what()));
        throw;
    }
}

Eigen::MatrixXd Layer::compute_regularization_gradient() const
{
    switch (regularization_type)
    {
    case RegularizationType::L1:
        return regularization_strength * weights.unaryExpr([](double x)
                                                           { return x > 0 ? 1.0 : -1.0; });
    case RegularizationType::L2:
        return regularization_strength * weights;
    default:
        return Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    }
}

double Layer::compute_regularization_loss() const
{
    switch (regularization_type)
    {
    case RegularizationType::L1:
        return regularization_strength * weights.array().abs().sum();
    case RegularizationType::L2:
        return 0.5 * regularization_strength * weights.array().square().sum();
    default:
        return 0.0;
    }
}

void Layer::set_learning_rate(double new_learning_rate)
{
    try
    {
        Logger::log("Setting new learning rate: " + std::to_string(new_learning_rate));
        learning_rate = new_learning_rate;
        optimizer->setLearningRate(new_learning_rate);
    }
    catch (const std::exception &e)
    {
        Logger::log_exception(e, "Layer::set_learning_rate");
        throw;
    }
}

void Layer::set_regularization(RegularizationType type, double strength)
{
    try
    {
        Logger::log("Setting regularization for layer");

        if (strength < 0)
        {
            throw std::invalid_argument("Regularization strength must be non-negative");
        }

        regularization_type = type;
        regularization_strength = strength;

        std::string reg_type_str;
        switch (type)
        {
        case RegularizationType::None:
            reg_type_str = "None";
            break;
        case RegularizationType::L1:
            reg_type_str = "L1";
            break;
        case RegularizationType::L2:
            reg_type_str = "L2";
            break;
        default:
            reg_type_str = "Unknown";
        }

        Logger::log("Regularization set - Type: " + reg_type_str + ", Strength: " + std::to_string(strength));
    }
    catch (const std::exception &e)
    {
        Logger::log_exception(e, "Layer::set_regularization");
        throw;
    }
}