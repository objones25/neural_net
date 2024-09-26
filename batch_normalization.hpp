#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include "exceptions.hpp"

class BatchNorm
{
public:
    struct BatchNormCache
    {
        Eigen::VectorXd x;
        std::vector<Eigen::VectorXd> mean;
        std::vector<Eigen::VectorXd> var;
        std::vector<Eigen::VectorXd> normalized;
    };

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

    std::pair<Eigen::VectorXd, BatchNormCache> forward(const Eigen::VectorXd &x, bool training = true) const
    {
        if (x.size() % features_ != 0) {
            throw BatchNormalizationError("Input size is not a multiple of the number of features");
        }

        Eigen::VectorXd result(x.size());
        BatchNormCache cache;
        cache.x = x;

        for (int i = 0; i < x.size(); i += features_)
        {
            Eigen::VectorXd slice = x.segment(i, features_);
            Eigen::VectorXd normalized;
            if (training)
            {
                Eigen::VectorXd mean = slice.mean() * Eigen::VectorXd::Ones(features_);
                Eigen::VectorXd var = (slice.array() - mean.array()).square().mean() * Eigen::VectorXd::Ones(features_);
                normalized = (slice.array() - mean.array()) / (var.array() + epsilon_).sqrt();

                cache.mean.push_back(mean);
                cache.var.push_back(var);
                cache.normalized.push_back(normalized);
            }
            else
            {
                normalized = (slice.array() - running_mean_.array()) / (running_var_.array() + epsilon_).sqrt();
            }
            result.segment(i, features_) = gamma_.array() * normalized.array() + beta_.array();
        }

        if (training)
        {
            update_running_stats(cache.mean.back(), cache.var.back());
        }

        return {result, cache};
    }

    Eigen::VectorXd backward(const Eigen::VectorXd &dout, const BatchNormCache &cache)
    {
        if (dout.size() != cache.x.size()) {
            throw BatchNormalizationError("Input size mismatch in BatchNorm backward pass");
        }

        Eigen::VectorXd dx = Eigen::VectorXd::Zero(dout.size());
        gamma_grad_ = Eigen::VectorXd::Zero(features_);
        beta_grad_ = Eigen::VectorXd::Zero(features_);

        int num_batches = dout.size() / features_;

        for (int i = 0; i < dout.size(); i += features_)
        {
            Eigen::VectorXd slice = cache.x.segment(i, features_);
            Eigen::VectorXd dout_slice = dout.segment(i, features_);
            Eigen::VectorXd mean = cache.mean[i / features_];
            Eigen::VectorXd var = cache.var[i / features_];
            Eigen::VectorXd normalized = cache.normalized[i / features_];

            gamma_grad_ += dout_slice.cwiseProduct(normalized);
            beta_grad_ += dout_slice;

            Eigen::ArrayXd dx_norm = dout_slice.array() * gamma_.array();

            double dvar = (dx_norm * (slice.array() - mean.array()) * (-0.5) * (var.array() + epsilon_).pow(-1.5)).sum();

            double dmean = (dx_norm * (-1.0 / (var.array() + epsilon_).sqrt())).sum() +
                           dvar * (-2.0 * (slice.array() - mean.array()).sum()) / features_;

            dx.segment(i, features_) = (dx_norm / (var.array() + epsilon_).sqrt() +
                                        dvar * 2.0 * (slice.array() - mean.array()) / features_ +
                                        dmean / features_).matrix();
        }

        gamma_grad_ /= num_batches;
        beta_grad_ /= num_batches;

        return dx;
    }

    void update_running_stats(const Eigen::VectorXd &mean, const Eigen::VectorXd &var) const
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
    double get_epsilon() const { return epsilon_; }

private:
    int features_;
    Eigen::VectorXd gamma_, beta_;
    mutable Eigen::VectorXd running_mean_, running_var_;
    Eigen::VectorXd gamma_grad_, beta_grad_;
    double momentum_, epsilon_;
};