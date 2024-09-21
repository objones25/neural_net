#pragma once
#include <Eigen/Dense>

class BatchNorm {
public:
    BatchNorm(int features) : gamma(Eigen::VectorXd::Ones(features)),
                              beta(Eigen::VectorXd::Zero(features)),
                              running_mean(Eigen::VectorXd::Zero(features)),
                              running_var(Eigen::VectorXd::Ones(features)),
                              momentum(0.1), epsilon(1e-8) {}

    Eigen::VectorXd forward(const Eigen::VectorXd& x, bool training = true) const {
        if (training) {
            double mean = x.mean();
            double var = ((x.array() - mean).square().sum()) / (x.size() - 1);
            
            Eigen::VectorXd x_norm = (x.array() - mean) / std::sqrt(var + epsilon);
            return gamma.array() * x_norm.array() + beta.array();
        } else {
            return gamma.array() * ((x.array() - running_mean.array()) / (running_var.array() + epsilon).sqrt()) + beta.array();
        }
    }

    Eigen::VectorXd backward(const Eigen::VectorXd& dout, const Eigen::VectorXd& x_norm) const {
        Eigen::VectorXd dgamma = (dout.array() * x_norm.array()).matrix();
        Eigen::VectorXd dbeta = dout;
        Eigen::VectorXd dx_norm = dout.array() * gamma.array();
        
        // Simplified backward pass, you may need to expand this
        return dx_norm;
    }

    void update_parameters(const Eigen::VectorXd& d_gamma, const Eigen::VectorXd& d_beta, double learning_rate) {
        gamma -= learning_rate * d_gamma;
        beta -= learning_rate * d_beta;
    }

    void update_running_stats(const Eigen::VectorXd& batch_mean, const Eigen::VectorXd& batch_var) {
        running_mean = momentum * running_mean + (1 - momentum) * batch_mean;
        running_var = momentum * running_var + (1 - momentum) * batch_var;
    }

private:
    Eigen::VectorXd gamma, beta;
    Eigen::VectorXd running_mean, running_var;
    double momentum, epsilon;
};