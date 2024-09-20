#pragma once

#include <Eigen/Dense>
#include <memory>

class OptimizationAlgorithm {
public:
    virtual void update(Eigen::MatrixXd& w, Eigen::VectorXd& b,
                        const Eigen::MatrixXd& dw, const Eigen::VectorXd& db) = 0;
    virtual ~OptimizationAlgorithm() = default;
};

class GradientDescent : public OptimizationAlgorithm {
private:
    double learning_rate;

public:
    GradientDescent(double lr);
    void update(Eigen::MatrixXd& w, Eigen::VectorXd& b,
                const Eigen::MatrixXd& dw, const Eigen::VectorXd& db) override;
};

class Adam : public OptimizationAlgorithm {
private:
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    int t;
    Eigen::MatrixXd m_w, v_w;
    Eigen::VectorXd m_b, v_b;

public:
    Adam(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8);
    void update(Eigen::MatrixXd& w, Eigen::VectorXd& b,
                const Eigen::MatrixXd& dw, const Eigen::VectorXd& db) override;
};

class RMSprop : public OptimizationAlgorithm {
private:
    double learning_rate;
    double beta;
    double epsilon;
    Eigen::MatrixXd v_w;
    Eigen::VectorXd v_b;

public:
    RMSprop(double lr = 0.001, double b = 0.9, double eps = 1e-8);
    void update(Eigen::MatrixXd& w, Eigen::VectorXd& b,
                const Eigen::MatrixXd& dw, const Eigen::VectorXd& db) override;
};

// Factory function to create optimizers
std::unique_ptr<OptimizationAlgorithm> create_optimizer(const std::string& name, double learning_rate);