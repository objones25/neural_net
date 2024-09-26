#include "optimization_algorithms.hpp"

GradientDescent::GradientDescent(double lr) : learning_rate(lr) {}

void GradientDescent::update(Layer &layer, const Eigen::MatrixXd &dw, const Eigen::VectorXd &db)
{
    if (dw.rows() != layer.weights.rows() || dw.cols() != layer.weights.cols() ||
        db.size() != layer.biases.size())
    {
        throw OptimizerError("Gradient dimensions do not match layer dimensions");
    }

    layer.weights -= learning_rate * dw;
    layer.biases -= learning_rate * db;

    if (layer.batch_norm)
    {
        layer.batch_norm->set_gamma(layer.batch_norm->get_gamma() - learning_rate * layer.bn_gamma_grad);
        layer.batch_norm->set_beta(layer.batch_norm->get_beta() - learning_rate * layer.bn_beta_grad);
    }
}

Adam::Adam(double lr, double b1, double b2, double eps)
    : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

void Adam::update(Layer &layer, const Eigen::MatrixXd &dw, const Eigen::VectorXd &db)
{
    if (dw.rows() != layer.weights.rows() || dw.cols() != layer.weights.cols() ||
        db.size() != layer.biases.size())
    {
        throw OptimizerError("Gradient dimensions do not match layer dimensions");
    }

    t++;

    // Initialize momentum and velocity if not already done
    if (m_w.rows() == 0)
    {
        m_w = Eigen::MatrixXd::Zero(layer.weights.rows(), layer.weights.cols());
        v_w = Eigen::MatrixXd::Zero(layer.weights.rows(), layer.weights.cols());
        m_b = Eigen::VectorXd::Zero(layer.biases.size());
        v_b = Eigen::VectorXd::Zero(layer.biases.size());
    }

    // Update biased first moment estimate
    m_w = beta1 * m_w + (1.0 - beta1) * dw;
    v_w = beta2 * v_w + (1.0 - beta2) * dw.array().square().matrix();
    m_b = beta1 * m_b + (1.0 - beta1) * db;
    v_b = beta2 * v_b + (1.0 - beta2) * db.array().square().matrix();

    // Compute bias-corrected first and second moment estimates
    Eigen::MatrixXd m_w_hat = m_w / (1.0 - std::pow(beta1, t));
    Eigen::MatrixXd v_w_hat = v_w / (1.0 - std::pow(beta2, t));
    Eigen::VectorXd m_b_hat = m_b / (1.0 - std::pow(beta1, t));
    Eigen::VectorXd v_b_hat = v_b / (1.0 - std::pow(beta2, t));

    // Update parameters
    layer.weights -= (learning_rate * m_w_hat.array() / (v_w_hat.array().sqrt() + epsilon)).matrix();
    layer.biases -= (learning_rate * m_b_hat.array() / (v_b_hat.array().sqrt() + epsilon)).matrix();

    if (layer.batch_norm)
    {
        Eigen::VectorXd m_gamma_hat = layer.bn_gamma_grad / (1.0 - std::pow(beta1, t));
        Eigen::VectorXd v_gamma_hat = layer.bn_gamma_grad.array().square() / (1.0 - std::pow(beta2, t));
        Eigen::VectorXd m_beta_hat = layer.bn_beta_grad / (1.0 - std::pow(beta1, t));
        Eigen::VectorXd v_beta_hat = layer.bn_beta_grad.array().square() / (1.0 - std::pow(beta2, t));

        layer.batch_norm->set_gamma(layer.batch_norm->get_gamma() -
                                    (learning_rate * m_gamma_hat.array() / (v_gamma_hat.array().sqrt() + epsilon)).matrix());
        layer.batch_norm->set_beta(layer.batch_norm->get_beta() -
                                   (learning_rate * m_beta_hat.array() / (v_beta_hat.array().sqrt() + epsilon)).matrix());
    }
}

RMSprop::RMSprop(double lr, double decay, double eps)
    : learning_rate(lr), decay_rate(decay), epsilon(eps) {}

void RMSprop::update(Layer &layer, const Eigen::MatrixXd &dw, const Eigen::VectorXd &db)
{
    if (dw.rows() != layer.weights.rows() || dw.cols() != layer.weights.cols() ||
        db.size() != layer.biases.size())
    {
        throw OptimizerError("Gradient dimensions do not match layer dimensions in RMSprop");
    }

    // Initialize square gradients if not already done
    if (square_grad_w.rows() == 0)
    {
        square_grad_w = Eigen::MatrixXd::Zero(layer.weights.rows(), layer.weights.cols());
        square_grad_b = Eigen::VectorXd::Zero(layer.biases.size());
    }

    // Update square gradients
    square_grad_w = decay_rate * square_grad_w + (1 - decay_rate) * dw.array().square().matrix();
    square_grad_b = decay_rate * square_grad_b + (1 - decay_rate) * db.array().square().matrix();

    // Update weights and biases
    layer.weights -= (learning_rate * dw.array() / (square_grad_w.array().sqrt() + epsilon)).matrix();
    layer.biases -= (learning_rate * db.array() / (square_grad_b.array().sqrt() + epsilon)).matrix();

    // Update batch normalization parameters if present
    if (layer.batch_norm)
    {
        Eigen::VectorXd square_grad_gamma = layer.bn_gamma_grad.array().square().matrix();
        Eigen::VectorXd square_grad_beta = layer.bn_beta_grad.array().square().matrix();

        layer.batch_norm->set_gamma(layer.batch_norm->get_gamma() -
                                    (learning_rate * layer.bn_gamma_grad.array() / (square_grad_gamma.array().sqrt() + epsilon)).matrix());
        layer.batch_norm->set_beta(layer.batch_norm->get_beta() -
                                   (learning_rate * layer.bn_beta_grad.array() / (square_grad_beta.array().sqrt() + epsilon)).matrix());
    }
}

std::unique_ptr<OptimizationAlgorithm> create_optimizer(const std::string &name, double learning_rate)
{
    if (name == "GradientDescent")
    {
        return std::make_unique<GradientDescent>(learning_rate);
    }
    else if (name == "Adam")
    {
        return std::make_unique<Adam>(learning_rate);
    }
    else if (name == "RMSprop")
    {
        return std::make_unique<RMSprop>(learning_rate);
    }
    else
    {
        throw std::invalid_argument("Unknown optimizer: " + name);
    }
}