#include "optimization_algorithms.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>

using namespace Eigen;

void update_weights_and_biases(OptimizationAlgorithm algo,
                               MatrixXd &w, VectorXd &b,
                               const MatrixXd &dw, const VectorXd &db,
                               MatrixXd &m_w, VectorXd &m_b,
                               MatrixXd &v_w, VectorXd &v_b,
                               double learning_rate, double beta1, double beta2, double epsilon,
                               int &t)
{
    t++;

    switch (algo)
    {
    case OptimizationAlgorithm::GradientDescent:
    {
        w -= learning_rate * dw;
        b -= learning_rate * db;
    }
    break;

    case OptimizationAlgorithm::Adam:
    {
        // Initialize m_w and v_w if they're empty
        if (m_w.size() == 0)
            m_w = MatrixXd::Zero(w.rows(), w.cols());
        if (v_w.size() == 0)
            v_w = MatrixXd::Zero(w.rows(), w.cols());
        if (m_b.size() == 0)
            m_b = VectorXd::Zero(b.size());
        if (v_b.size() == 0)
            v_b = VectorXd::Zero(b.size());

        m_w = beta1 * m_w + (1.0 - beta1) * dw;
        m_b = beta1 * m_b + (1.0 - beta1) * db;

        v_w = beta2 * v_w.array() + (1.0 - beta2) * dw.array().square();
        v_b = beta2 * v_b.array() + (1.0 - beta2) * db.array().square();

        double bias_correction1 = 1.0 - std::pow(beta1, t);
        double bias_correction2 = 1.0 - std::pow(beta2, t);

        MatrixXd m_w_hat = m_w / bias_correction1;
        VectorXd m_b_hat = m_b / bias_correction1;
        MatrixXd v_w_hat = v_w / bias_correction2;
        VectorXd v_b_hat = v_b / bias_correction2;

        w -= (learning_rate * m_w_hat.array() / (v_w_hat.array().sqrt() + epsilon)).matrix();
        b -= (learning_rate * m_b_hat.array() / (v_b_hat.array().sqrt() + epsilon)).matrix();
    }
    break;

    case OptimizationAlgorithm::RMSprop:
    {
        // Initialize v_w and v_b if they're empty
        if (v_w.size() == 0)
            v_w = MatrixXd::Constant(w.rows(), w.cols(), epsilon);
        if (v_b.size() == 0)
            v_b = VectorXd::Constant(b.size(), epsilon);

        // Update accumulated squared gradients
        v_w = beta2 * v_w + (1.0 - beta2) * dw.array().square().matrix();
        v_b = beta2 * v_b + (1.0 - beta2) * db.array().square().matrix();

        // Update parameters
        w -= learning_rate * (dw.array() / v_w.array().sqrt()).matrix();
        b -= learning_rate * (db.array() / v_b.array().sqrt()).matrix();
    }
    break;

    default:
        throw std::runtime_error("Unknown optimization algorithm");
    }
}