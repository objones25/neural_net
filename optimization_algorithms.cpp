#include "optimization_algorithms.hpp"
#include <cmath>
#include <stdexcept>

// Gradient Descent optimizer
GradientDescent::GradientDescent(double lr) : learning_rate(lr) {}

void GradientDescent::update(Eigen::MatrixXd& w, Eigen::VectorXd& b,
                             const Eigen::MatrixXd& dw, const Eigen::VectorXd& db) {
    // Update weights and biases using the gradients and learning rate
    w -= learning_rate * dw;
    b -= learning_rate * db;
}

// Adam optimizer
Adam::Adam(double lr, double b1, double b2, double eps)
    : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

void Adam::update(Eigen::MatrixXd& w, Eigen::VectorXd& b,
                  const Eigen::MatrixXd& dw, const Eigen::VectorXd& db) {
    t++;

    // Initialize momentum and velocity if not already done
    if (m_w.rows() == 0) {
        m_w = Eigen::MatrixXd::Zero(w.rows(), w.cols());
        v_w = Eigen::MatrixXd::Zero(w.rows(), w.cols());
        m_b = Eigen::VectorXd::Zero(b.size());
        v_b = Eigen::VectorXd::Zero(b.size());
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
    double lr_t = learning_rate * std::sqrt(1.0 - std::pow(beta2, t)) / (1.0 - std::pow(beta1, t));
    w -= (lr_t * m_w_hat.array() / (v_w_hat.array().sqrt() + epsilon)).matrix();
    b -= (lr_t * m_b_hat.array() / (v_b_hat.array().sqrt() + epsilon)).matrix();
}
// RMSprop optimizer
RMSprop::RMSprop(double lr, double b, double eps)
    : learning_rate(lr), beta(b), epsilon(eps) {}

void RMSprop::update(Eigen::MatrixXd& w, Eigen::VectorXd& b,
                     const Eigen::MatrixXd& dw, const Eigen::VectorXd& db) {
    // Initialize moving average of squared gradients if not already done
    if (v_w.rows() == 0) {
        v_w = Eigen::MatrixXd::Zero(w.rows(), w.cols());
        v_b = Eigen::VectorXd::Zero(b.size());
    }

    // Update moving average of squared gradients
    v_w = beta * v_w + (1.0 - beta) * dw.array().square().matrix();
    v_b = beta * v_b + (1.0 - beta) * db.array().square().matrix();

    // Update parameters
    w -= (learning_rate * dw.array() / (v_w.array().sqrt() + epsilon)).matrix();
    b -= (learning_rate * db.array() / (v_b.array().sqrt() + epsilon)).matrix();
}

// Factory function to create optimizer based on name
std::unique_ptr<OptimizationAlgorithm> create_optimizer(const std::string& name, double learning_rate) {
    if (name == "GradientDescent") {
        return std::make_unique<GradientDescent>(learning_rate);
    } else if (name == "Adam") {
        return std::make_unique<Adam>(learning_rate);
    } else if (name == "RMSprop") {
        return std::make_unique<RMSprop>(learning_rate);
    } else {
        throw std::invalid_argument("Unknown optimizer name: " + name);
    }
}