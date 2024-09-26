#pragma once
#include <Eigen/Dense>
#include "layer.hpp"
#include "exceptions.hpp"

class OptimizationAlgorithm {
public:
    virtual void update(Layer& layer, const Eigen::MatrixXd& dw, const Eigen::VectorXd& db) = 0;
    virtual void setLearningRate(double lr) = 0;
    virtual double getLearningRate() const = 0;
    virtual ~OptimizationAlgorithm() = default;
};

class GradientDescent : public OptimizationAlgorithm {
private:
    double learning_rate;

public:
    GradientDescent(double lr);
    void update(Layer& layer, const Eigen::MatrixXd& dw, const Eigen::VectorXd& db) override;
    void setLearningRate(double lr) override { learning_rate = lr; }
    double getLearningRate() const override { return learning_rate; }
};

class Adam : public OptimizationAlgorithm {
private:
    double learning_rate;
    double beta1, beta2, epsilon;
    int t;
    Eigen::MatrixXd m_w, v_w;
    Eigen::VectorXd m_b, v_b;

public:
    Adam(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8);
    void update(Layer& layer, const Eigen::MatrixXd& dw, const Eigen::VectorXd& db) override;
    void setLearningRate(double lr) override { learning_rate = lr; }
    double getLearningRate() const override { return learning_rate; }
};

class RMSprop : public OptimizationAlgorithm {
private:
    double learning_rate;
    double decay_rate;
    double epsilon;
    Eigen::MatrixXd square_grad_w;
    Eigen::VectorXd square_grad_b;

public:
    RMSprop(double lr = 0.001, double decay = 0.9, double eps = 1e-8);
    void update(Layer& layer, const Eigen::MatrixXd& dw, const Eigen::VectorXd& db) override;
    void setLearningRate(double lr) override { learning_rate = lr; }
    double getLearningRate() const override { return learning_rate; }
};

std::unique_ptr<OptimizationAlgorithm> create_optimizer(const std::string& name, double learning_rate);