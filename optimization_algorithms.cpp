#include "optimization_algorithms.hpp"
#include <cmath>
#include <limits>

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

    // Add safeguards
    auto clip_and_check = [](Eigen::MatrixXd& mat, const std::string& name) {
        double max_val = 1e6;
        mat = mat.cwiseMin(max_val).cwiseMax(-max_val);
        if (!mat.allFinite()) {
            throw NumericalInstabilityError("Non-finite values detected in " + name);
        }
    };

    auto clip_and_check_vector = [](Eigen::VectorXd& vec, const std::string& name) {
        double max_val = 1e6;
        vec = vec.cwiseMin(max_val).cwiseMax(-max_val);
        if (!vec.allFinite()) {
            throw NumericalInstabilityError("Non-finite values detected in " + name);
        }
    };

    clip_and_check(m_w_hat, "Adam m_w_hat");
    clip_and_check(v_w_hat, "Adam v_w_hat");
    clip_and_check_vector(m_b_hat, "Adam m_b_hat");
    clip_and_check_vector(v_b_hat, "Adam v_b_hat");

    // Update parameters
    Eigen::MatrixXd update_w = (learning_rate * m_w_hat.array() / (v_w_hat.array().sqrt() + epsilon)).matrix();
    Eigen::VectorXd update_b = (learning_rate * m_b_hat.array() / (v_b_hat.array().sqrt() + epsilon)).matrix();

    clip_and_check(update_w, "Adam weight update");
    clip_and_check_vector(update_b, "Adam bias update");

    layer.weights -= update_w;
    layer.biases -= update_b;

    if (layer.batch_norm)
    {
        Eigen::VectorXd m_gamma_hat = layer.bn_gamma_grad / (1.0 - std::pow(beta1, t));
        Eigen::VectorXd v_gamma_hat = layer.bn_gamma_grad.array().square() / (1.0 - std::pow(beta2, t));
        Eigen::VectorXd m_beta_hat = layer.bn_beta_grad / (1.0 - std::pow(beta1, t));
        Eigen::VectorXd v_beta_hat = layer.bn_beta_grad.array().square() / (1.0 - std::pow(beta2, t));

        clip_and_check_vector(m_gamma_hat, "Adam m_gamma_hat");
        clip_and_check_vector(v_gamma_hat, "Adam v_gamma_hat");
        clip_and_check_vector(m_beta_hat, "Adam m_beta_hat");
        clip_and_check_vector(v_beta_hat, "Adam v_beta_hat");

        Eigen::VectorXd update_gamma = (learning_rate * m_gamma_hat.array() / (v_gamma_hat.array().sqrt() + epsilon)).matrix();
        Eigen::VectorXd update_beta = (learning_rate * m_beta_hat.array() / (v_beta_hat.array().sqrt() + epsilon)).matrix();

        clip_and_check_vector(update_gamma, "Adam gamma update");
        clip_and_check_vector(update_beta, "Adam beta update");

        layer.batch_norm->set_gamma(layer.batch_norm->get_gamma() - update_gamma);
        layer.batch_norm->set_beta(layer.batch_norm->get_beta() - update_beta);
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

    // Add safeguards
    auto clip_and_check = [](Eigen::MatrixXd& mat, const std::string& name) {
        double max_val = 1e6;
        mat = mat.cwiseMin(max_val).cwiseMax(-max_val);
        if (!mat.allFinite()) {
            throw NumericalInstabilityError("Non-finite values detected in " + name);
        }
    };

    auto clip_and_check_vector = [](Eigen::VectorXd& vec, const std::string& name) {
        double max_val = 1e6;
        vec = vec.cwiseMin(max_val).cwiseMax(-max_val);
        if (!vec.allFinite()) {
            throw NumericalInstabilityError("Non-finite values detected in " + name);
        }
    };

    clip_and_check(square_grad_w, "RMSprop square_grad_w");
    clip_and_check_vector(square_grad_b, "RMSprop square_grad_b");

    // Update weights and biases
    Eigen::MatrixXd update_w = (learning_rate * dw.array() / (square_grad_w.array().sqrt() + epsilon)).matrix();
    Eigen::VectorXd update_b = (learning_rate * db.array() / (square_grad_b.array().sqrt() + epsilon)).matrix();

    clip_and_check(update_w, "RMSprop weight update");
    clip_and_check_vector(update_b, "RMSprop bias update");

    layer.weights -= update_w;
    layer.biases -= update_b;

    // Update batch normalization parameters if present
    if (layer.batch_norm)
    {
        Eigen::VectorXd square_grad_gamma = layer.bn_gamma_grad.array().square().matrix();
        Eigen::VectorXd square_grad_beta = layer.bn_beta_grad.array().square().matrix();

        clip_and_check_vector(square_grad_gamma, "RMSprop square_grad_gamma");
        clip_and_check_vector(square_grad_beta, "RMSprop square_grad_beta");

        Eigen::VectorXd update_gamma = (learning_rate * layer.bn_gamma_grad.array() / (square_grad_gamma.array().sqrt() + epsilon)).matrix();
        Eigen::VectorXd update_beta = (learning_rate * layer.bn_beta_grad.array() / (square_grad_beta.array().sqrt() + epsilon)).matrix();

        clip_and_check_vector(update_gamma, "RMSprop gamma update");
        clip_and_check_vector(update_beta, "RMSprop beta update");

        layer.batch_norm->set_gamma(layer.batch_norm->get_gamma() - update_gamma);
        layer.batch_norm->set_beta(layer.batch_norm->get_beta() - update_beta);
    }
}

void GradientDescent::setLearningRate(double lr) { learning_rate = lr; }
double GradientDescent::getLearningRate() const { return learning_rate; }

void Adam::setLearningRate(double lr) { learning_rate = lr; }
double Adam::getLearningRate() const { return learning_rate; }

void RMSprop::setLearningRate(double lr) { learning_rate = lr; }
double RMSprop::getLearningRate() const { return learning_rate; }

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