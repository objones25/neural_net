#pragma once
#include <Eigen/Dense>
#include <cmath>

class BatchNorm {
public:
    BatchNorm(int features, double momentum = 0.1, double epsilon = 1e-5)
        : features_(features),
          gamma_(Eigen::VectorXd::Ones(features)),
          beta_(Eigen::VectorXd::Zero(features)),
          running_mean_(Eigen::VectorXd::Zero(features)),
          running_var_(Eigen::VectorXd::Ones(features)),
          momentum_(momentum),
          epsilon_(epsilon) {}

    Eigen::VectorXd forward(const Eigen::VectorXd& x, bool training = true) const {
        if (training) {
            Eigen::VectorXd mean = x.array().head(features_).mean() * Eigen::VectorXd::Ones(features_);
            Eigen::VectorXd var = ((x.array().head(features_) - mean.array()).square().mean()) * Eigen::VectorXd::Ones(features_);
            
            Eigen::VectorXd x_norm = (x.array().head(features_) - mean.array()) / (var.array() + epsilon_).sqrt();
            return (gamma_.array() * x_norm.array() + beta_.array()).replicate((x.size() + features_ - 1) / features_, 1);
        } else {
            Eigen::VectorXd x_norm = (x.array().head(features_) - running_mean_.array()) / (running_var_.array() + epsilon_).sqrt();
            return (gamma_.array() * x_norm.array() + beta_.array()).replicate((x.size() + features_ - 1) / features_, 1);
        }
    }

    Eigen::VectorXd backward(const Eigen::VectorXd& dout, const Eigen::VectorXd& x) {
        int N = features_;
        Eigen::VectorXd mean = x.array().head(features_).mean() * Eigen::VectorXd::Ones(features_);
        Eigen::VectorXd var = ((x.array().head(features_) - mean.array()).square().mean()) * Eigen::VectorXd::Ones(features_);
        Eigen::VectorXd x_norm = (x.array().head(features_) - mean.array()) / (var.array() + epsilon_).sqrt();

        Eigen::VectorXd dgamma = (dout.array().head(features_) * x_norm.array()).matrix();
        Eigen::VectorXd dbeta = dout.head(features_);

        Eigen::VectorXd dx_norm = dout.array().head(features_) * gamma_.array();
        Eigen::VectorXd dvar = (dx_norm.array() * (x.array().head(features_) - mean.array()) * -0.5 * (var.array() + epsilon_).pow(-1.5)).sum() * Eigen::VectorXd::Ones(features_);
        Eigen::VectorXd dmean = (dx_norm.array() * -1.0 / (var.array() + epsilon_).sqrt()).sum() * Eigen::VectorXd::Ones(features_);

        Eigen::VectorXd dx = dx_norm.array() / (var.array() + epsilon_).sqrt()
                             + dvar.array() * 2.0 * (x.array().head(features_) - mean.array()) / N
                             + dmean.array() / N;

        gamma_ -= dgamma;
        beta_ -= dbeta;

        return dx.replicate((x.size() + features_ - 1) / features_, 1);
    }

    void update_running_stats(const Eigen::VectorXd& mean, const Eigen::VectorXd& var) {
        running_mean_ = momentum_ * running_mean_ + (1.0 - momentum_) * mean;
        running_var_ = momentum_ * running_var_ + (1.0 - momentum_) * var;
    }

    int get_features() const { return features_; }
    const Eigen::VectorXd& get_gamma() const { return gamma_; }
    const Eigen::VectorXd& get_beta() const { return beta_; }
    void set_gamma(const Eigen::VectorXd& gamma) { gamma_ = gamma; }
    void set_beta(const Eigen::VectorXd& beta) { beta_ = beta; }

private:
    int features_;
    Eigen::VectorXd gamma_, beta_;
    Eigen::VectorXd running_mean_, running_var_;
    double momentum_, epsilon_;
};