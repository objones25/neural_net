#pragma once
#include <Eigen/Dense>
#include <cmath>

class BatchNorm
{
public:
    BatchNorm(int features, double momentum = 0.1, double epsilon = 1e-5)
        : features_(features),
          gamma_(Eigen::VectorXd::Ones(features)),
          beta_(Eigen::VectorXd::Zero(features)),
          running_mean_(Eigen::VectorXd::Zero(features)),
          running_var_(Eigen::VectorXd::Ones(features)),
          gamma_grad_(Eigen::VectorXd::Zero(features)),
          beta_grad_(Eigen::VectorXd::Zero(features)),
          momentum_(momentum),
          epsilon_(epsilon) {}

    Eigen::VectorXd forward(const Eigen::VectorXd &x, bool training = true) const
    {
        if (x.size() % features_ != 0)
        {
            throw std::runtime_error("Input size is not a multiple of the number of features in BatchNorm");
        }

        Eigen::VectorXd result(x.size());
        for (int i = 0; i < x.size(); i += features_)
        {
            Eigen::VectorXd slice = x.segment(i, features_);
            Eigen::VectorXd normalized;
            if (training)
            {
                Eigen::VectorXd mean = slice.mean() * Eigen::VectorXd::Ones(features_);
                Eigen::VectorXd var = (slice.array() - mean.array()).square().mean() * Eigen::VectorXd::Ones(features_);
                normalized = (slice.array() - mean.array()) / (var.array() + epsilon_).sqrt();
            }
            else
            {
                normalized = (slice.array() - running_mean_.array()) / (running_var_.array() + epsilon_).sqrt();
            }
            result.segment(i, features_) = gamma_.array() * normalized.array() + beta_.array();
        }
        return result;
    }

    Eigen::VectorXd backward(const Eigen::VectorXd &dout, const Eigen::VectorXd &x)
    {
        if (x.size() % features_ != 0 || dout.size() != x.size())
        {
            throw std::runtime_error("Input size mismatch in BatchNorm backward pass");
        }

        Eigen::VectorXd dx = Eigen::VectorXd::Zero(x.size());
        gamma_grad_ = Eigen::VectorXd::Zero(features_);
        beta_grad_ = Eigen::VectorXd::Zero(features_);

        int num_batches = x.size() / features_;

        for (int i = 0; i < x.size(); i += features_)
        {
            Eigen::VectorXd slice = x.segment(i, features_);
            Eigen::VectorXd dout_slice = dout.segment(i, features_);

            double mean = slice.mean();
            Eigen::VectorXd centered = slice.array() - mean;
            double var = centered.array().square().mean();
            double std_dev = std::sqrt(var + epsilon_);
            Eigen::VectorXd x_norm = centered / std_dev;

            gamma_grad_ += dout_slice.cwiseProduct(x_norm);
            beta_grad_ += dout_slice;

            Eigen::VectorXd dx_norm = dout_slice.cwiseProduct(gamma_);

            Eigen::VectorXd dvar = (dx_norm.cwiseProduct(centered) * (-0.5) * std::pow(var + epsilon_, -1.5)).sum() * Eigen::VectorXd::Ones(features_);

            Eigen::VectorXd dmean = (-dx_norm.sum() / std_dev) * Eigen::VectorXd::Ones(features_) +
                                    dvar * (-2.0 * centered.sum()) / features_;

            dx.segment(i, features_) = dx_norm / std_dev +
                                       (dvar.array() * 2 * centered.array() / features_).matrix() +
                                       dmean / features_;
        }

        gamma_grad_ /= num_batches;
        beta_grad_ /= num_batches;

        return dx;
    }

    void update_running_stats(const Eigen::VectorXd &mean, const Eigen::VectorXd &var)
    {
        running_mean_ = momentum_ * running_mean_ + (1.0 - momentum_) * mean;
        running_var_ = momentum_ * running_var_ + (1.0 - momentum_) * var;
    }

    int get_features() const { return features_; }
    const Eigen::VectorXd &get_gamma() const { return gamma_; }
    const Eigen::VectorXd &get_beta() const { return beta_; }
    void set_gamma(const Eigen::VectorXd &gamma) { gamma_ = gamma; }
    void set_beta(const Eigen::VectorXd &beta) { beta_ = beta; }
    const Eigen::VectorXd &get_gamma_grad() const { return gamma_grad_; }
    const Eigen::VectorXd &get_beta_grad() const { return beta_grad_; }

private:
    int features_;
    Eigen::VectorXd gamma_, beta_;
    Eigen::VectorXd running_mean_, running_var_;
    Eigen::VectorXd gamma_grad_, beta_grad_;
    double momentum_, epsilon_;
};