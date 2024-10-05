#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <string>
#include <stdexcept>
#include <limits>
#include "exceptions.hpp"
#include "logger.hpp"

class Optimizer
{
public:
    Optimizer(double learning_rate) : learning_rate(learning_rate)
    {
        if (learning_rate <= 0)
        {
            throw std::invalid_argument("Learning rate must be positive");
        }
    }
    virtual ~Optimizer() = default;

    virtual void update(Eigen::MatrixXd &weights, Eigen::VectorXd &biases,
                        const Eigen::MatrixXd &dw, const Eigen::VectorXd &db) = 0;

    void setLearningRate(double lr)
    {
        if (lr <= 0)
        {
            throw std::invalid_argument("Learning rate must be positive");
        }
        learning_rate = lr;
    }
    double getLearningRate() const { return learning_rate; }

protected:
    double learning_rate;

    void checkFinite(const Eigen::MatrixXd &mat, const std::string &name)
    {
        if (!mat.allFinite())
        {
            Logger::error("Non-finite values detected in " + name);
            throw NumericalInstabilityError("Non-finite values detected in " + name);
        }
    }

    void checkFinite(const Eigen::VectorXd &vec, const std::string &name)
    {
        if (!vec.allFinite())
        {
            Logger::error("Non-finite values detected in " + name);
            throw NumericalInstabilityError("Non-finite values detected in " + name);
        }
    }

    template <typename Derived>
    void clipValues(Eigen::MatrixBase<Derived> &mat, double min_val, double max_val)
    {
        mat = mat.cwiseMax(min_val).cwiseMin(max_val);
    }
};

class RMSprop : public Optimizer
{
public:
    RMSprop(double learning_rate, double decay_rate = 0.9, double epsilon = 1e-8)
        : Optimizer(learning_rate), decay_rate(decay_rate), epsilon(epsilon)
    {
        if (decay_rate <= 0 || decay_rate >= 1)
        {
            throw std::invalid_argument("Decay rate must be between 0 and 1");
        }
        if (epsilon <= 0)
        {
            throw std::invalid_argument("Epsilon must be positive");
        }
    }

    void update(Eigen::MatrixXd &weights, Eigen::VectorXd &biases,
                const Eigen::MatrixXd &dw, const Eigen::VectorXd &db) override
    {
        try
        {
            Logger::log("RMSprop: Starting update", LogLevel::DEBUG);
            checkFinite(dw, "dw");
            checkFinite(db, "db");

            if (square_grad_w.rows() == 0)
            {
                square_grad_w = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
                square_grad_b = Eigen::VectorXd::Zero(biases.size());
            }

            // Update moving average of squared gradients
            square_grad_w = decay_rate * square_grad_w + (1 - decay_rate) * dw.array().square().matrix();
            square_grad_b = decay_rate * square_grad_b + (1 - decay_rate) * db.array().square().matrix();

            checkFinite(square_grad_w, "square_grad_w");
            checkFinite(square_grad_b, "square_grad_b");

            // Update parameters
            Eigen::MatrixXd weight_update = (learning_rate * dw.array() / (square_grad_w.array().sqrt() + epsilon)).matrix();
            Eigen::VectorXd bias_update = (learning_rate * db.array() / (square_grad_b.array().sqrt() + epsilon)).matrix();

            clipValues(weight_update, -1.0, 1.0); // Clip updates to prevent extreme values
            clipValues(bias_update, -1.0, 1.0);

            weights -= weight_update;
            biases -= bias_update;

            checkFinite(weights, "weights");
            checkFinite(biases, "biases");
            Logger::log("RMSprop: Finished update", LogLevel::DEBUG);
        }
        catch (const std::exception &e)
        {
            Logger::error("Error in RMSprop::update: " + std::string(e.what()));
            throw;
        }
    }

private:
    double decay_rate;
    double epsilon;
    Eigen::MatrixXd square_grad_w;
    Eigen::VectorXd square_grad_b;
};

class Adam : public Optimizer
{
public:
    Adam(double learning_rate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        : Optimizer(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0)
    {
        if (beta1 <= 0 || beta1 >= 1)
        {
            throw std::invalid_argument("Beta1 must be between 0 and 1");
        }
        if (beta2 <= 0 || beta2 >= 1)
        {
            throw std::invalid_argument("Beta2 must be between 0 and 1");
        }
        if (epsilon <= 0)
        {
            throw std::invalid_argument("Epsilon must be positive");
        }
    }

    void update(Eigen::MatrixXd &weights, Eigen::VectorXd &biases,
                const Eigen::MatrixXd &dw, const Eigen::VectorXd &db) override
    {
        try
        {
            Logger::log("Adam: Starting update", LogLevel::DEBUG);
            checkFinite(dw, "dw");
            checkFinite(db, "db");

            t++;

            if (m_w.rows() == 0)
            {
                m_w = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
                v_w = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
                m_b = Eigen::VectorXd::Zero(biases.size());
                v_b = Eigen::VectorXd::Zero(biases.size());
            }

            // Update biased first moment estimate
            m_w = beta1 * m_w + (1.0 - beta1) * dw;
            v_w = beta2 * v_w + (1.0 - beta2) * dw.array().square().matrix();
            m_b = beta1 * m_b + (1.0 - beta1) * db;
            v_b = beta2 * v_b + (1.0 - beta2) * db.array().square().matrix();

            checkFinite(m_w, "m_w");
            checkFinite(v_w, "v_w");
            checkFinite(m_b, "m_b");
            checkFinite(v_b, "v_b");

            // Compute bias-corrected first and second moment estimates
            double bias_correction1 = 1.0 - std::pow(beta1, t);
            double bias_correction2 = 1.0 - std::pow(beta2, t);

            Eigen::MatrixXd m_w_hat = m_w / bias_correction1;
            Eigen::MatrixXd v_w_hat = v_w / bias_correction2;
            Eigen::VectorXd m_b_hat = m_b / bias_correction1;
            Eigen::VectorXd v_b_hat = v_b / bias_correction2;

            // Update parameters
            Eigen::MatrixXd weight_update = (learning_rate * m_w_hat.array() / (v_w_hat.array().sqrt() + epsilon)).matrix();
            Eigen::VectorXd bias_update = (learning_rate * m_b_hat.array() / (v_b_hat.array().sqrt() + epsilon)).matrix();

            clipValues(weight_update, -1.0, 1.0); // Clip updates to prevent extreme values
            clipValues(bias_update, -1.0, 1.0);

            weights -= weight_update;
            biases -= bias_update;

            checkFinite(weights, "weights");
            checkFinite(biases, "biases");
            Logger::log("Adam: Finished update", LogLevel::DEBUG);
        }
        catch (const std::exception &e)
        {
            Logger::error("Error in Adam::update: " + std::string(e.what()));
            throw;
        }
    }

private:
    double beta1, beta2, epsilon;
    int t;
    Eigen::MatrixXd m_w, v_w;
    Eigen::VectorXd m_b, v_b;
};

inline std::unique_ptr<Optimizer> create_optimizer(const std::string &name, double learning_rate)
{
    try
    {
        if (name == "RMSprop")
        {
            return std::make_unique<RMSprop>(learning_rate);
        }
        else if (name == "Adam")
        {
            return std::make_unique<Adam>(learning_rate);
        }
        else
        {
            throw std::invalid_argument("Unknown optimizer: " + name);
        }
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("Error creating optimizer: " + std::string(e.what()));
    }
}