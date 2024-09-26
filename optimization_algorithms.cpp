#include "optimization_algorithms.hpp"
#include <cmath>
#include <limits>

const double MAX_LEARNING_RATE = 1.0;
const double CLIP_VALUE = 1.0;

// template <typename Derived>
// void clip_and_check(Eigen::MatrixBase<Derived>& mat, const std::string& name, double clip_value = 1e6) {
//     mat = mat.cwiseMin(clip_value).cwiseMax(-clip_value);
//     if (!mat.allFinite()) {
//         throw NumericalInstabilityError("Non-finite values detected in " + name);
//     }
// }

GradientDescent::GradientDescent(double lr) : learning_rate(lr) {}

void GradientDescent::update(Layer &layer, const Eigen::MatrixXd &dw, const Eigen::VectorXd &db)
{
    Eigen::MatrixXd clipped_dw = dw.cwiseMin(CLIP_VALUE).cwiseMax(-CLIP_VALUE);
    Eigen::VectorXd clipped_db = db.cwiseMin(CLIP_VALUE).cwiseMax(-CLIP_VALUE);

    double effective_lr = std::min(learning_rate, MAX_LEARNING_RATE);
    layer.weights -= effective_lr * clipped_dw;
    layer.biases -= effective_lr * clipped_db;

    if (layer.batch_norm)
    {
        Eigen::VectorXd clipped_gamma_grad = layer.bn_gamma_grad.cwiseMin(CLIP_VALUE).cwiseMax(-CLIP_VALUE);
        Eigen::VectorXd clipped_beta_grad = layer.bn_beta_grad.cwiseMin(CLIP_VALUE).cwiseMax(-CLIP_VALUE);
        layer.batch_norm->set_gamma(layer.batch_norm->get_gamma() - effective_lr * clipped_gamma_grad);
        layer.batch_norm->set_beta(layer.batch_norm->get_beta() - effective_lr * clipped_beta_grad);
    }
}

Adam::Adam(double lr, double b1, double b2, double eps)
    : learning_rate(lr), beta1(b1), beta2(b2), epsilon(std::max(eps, 1e-8)), t(0) {}

void Adam::update(Layer &layer, const Eigen::MatrixXd &dw, const Eigen::VectorXd &db)
{
    t++;

    // Initialize momentum and velocity if not already done
    if (m_w.rows() == 0)
    {
        m_w = Eigen::MatrixXd::Zero(layer.weights.rows(), layer.weights.cols());
        v_w = Eigen::MatrixXd::Zero(layer.weights.rows(), layer.weights.cols());
        m_b = Eigen::VectorXd::Zero(layer.biases.size());
        v_b = Eigen::VectorXd::Zero(layer.biases.size());
    }

    // Clip gradients
    Eigen::MatrixXd clipped_dw = dw.cwiseMin(CLIP_VALUE).cwiseMax(-CLIP_VALUE);
    Eigen::VectorXd clipped_db = db.cwiseMin(CLIP_VALUE).cwiseMax(-CLIP_VALUE);

    // Update biased first moment estimate
    m_w = beta1 * m_w + (1.0 - beta1) * clipped_dw;
    v_w = beta2 * v_w + (1.0 - beta2) * clipped_dw.array().square().matrix();
    m_b = beta1 * m_b + (1.0 - beta1) * clipped_db;
    v_b = beta2 * v_b + (1.0 - beta2) * clipped_db.array().square().matrix();

    // Compute bias-corrected first and second moment estimates
    double bias_correction1 = 1.0 - std::pow(beta1, t);
    double bias_correction2 = 1.0 - std::pow(beta2, t);
    Eigen::MatrixXd m_w_hat = m_w / bias_correction1;
    Eigen::MatrixXd v_w_hat = v_w / bias_correction2;
    Eigen::VectorXd m_b_hat = m_b / bias_correction1;
    Eigen::VectorXd v_b_hat = v_b / bias_correction2;

    // Clip and check for numerical instability
    clip_and_check(m_w_hat, "Adam m_w_hat");
    clip_and_check(v_w_hat, "Adam v_w_hat");
    clip_and_check(m_b_hat, "Adam m_b_hat");
    clip_and_check(v_b_hat, "Adam v_b_hat");

    // Update parameters with a more numerically stable rule
    double effective_lr = std::min(learning_rate, MAX_LEARNING_RATE);
    layer.weights -= effective_lr * (m_w_hat.array() / (v_w_hat.array().sqrt() + epsilon).cwiseMax(epsilon)).matrix();
    layer.biases -= effective_lr * (m_b_hat.array() / (v_b_hat.array().sqrt() + epsilon).cwiseMax(epsilon)).matrix();

    if (layer.batch_norm)
    {
        Eigen::VectorXd clipped_gamma_grad = layer.bn_gamma_grad.cwiseMin(CLIP_VALUE).cwiseMax(-CLIP_VALUE);
        Eigen::VectorXd clipped_beta_grad = layer.bn_beta_grad.cwiseMin(CLIP_VALUE).cwiseMax(-CLIP_VALUE);

        // Initialize m_gamma, v_gamma, m_beta, and v_beta if not already done
        if (m_gamma.size() == 0)
        {
            m_gamma = Eigen::VectorXd::Zero(clipped_gamma_grad.size());
            v_gamma = Eigen::VectorXd::Zero(clipped_gamma_grad.size());
            m_beta = Eigen::VectorXd::Zero(clipped_beta_grad.size());
            v_beta = Eigen::VectorXd::Zero(clipped_beta_grad.size());
        }

        m_gamma = beta1 * m_gamma + (1.0 - beta1) * clipped_gamma_grad;
        v_gamma = beta2 * v_gamma + (1.0 - beta2) * clipped_gamma_grad.array().square().matrix();
        m_beta = beta1 * m_beta + (1.0 - beta1) * clipped_beta_grad;
        v_beta = beta2 * v_beta + (1.0 - beta2) * clipped_beta_grad.array().square().matrix();

        Eigen::VectorXd m_gamma_hat = m_gamma / bias_correction1;
        Eigen::VectorXd v_gamma_hat = v_gamma / bias_correction2;
        Eigen::VectorXd m_beta_hat = m_beta / bias_correction1;
        Eigen::VectorXd v_beta_hat = v_beta / bias_correction2;

        clip_and_check(m_gamma_hat, "Adam m_gamma_hat");
        clip_and_check(v_gamma_hat, "Adam v_gamma_hat");
        clip_and_check(m_beta_hat, "Adam m_beta_hat");
        clip_and_check(v_beta_hat, "Adam v_beta_hat");

        Eigen::VectorXd update_gamma = effective_lr * (m_gamma_hat.array() / (v_gamma_hat.array().sqrt() + epsilon).cwiseMax(epsilon)).matrix();
        Eigen::VectorXd update_beta = effective_lr * (m_beta_hat.array() / (v_beta_hat.array().sqrt() + epsilon).cwiseMax(epsilon)).matrix();

        clip_and_check(update_gamma, "Adam gamma update");
        clip_and_check(update_beta, "Adam beta update");

        layer.batch_norm->set_gamma(layer.batch_norm->get_gamma() - update_gamma);
        layer.batch_norm->set_beta(layer.batch_norm->get_beta() - update_beta);
    }
}

RMSprop::RMSprop(double lr, double decay, double eps)
    : learning_rate(lr), decay_rate(decay), epsilon(std::max(eps, 1e-8)) {}

void RMSprop::update(Layer &layer, const Eigen::MatrixXd &dw, const Eigen::VectorXd &db)
{
    // Initialize square gradients if not already done
    if (square_grad_w.rows() == 0)
    {
        square_grad_w = Eigen::MatrixXd::Zero(layer.weights.rows(), layer.weights.cols());
        square_grad_b = Eigen::VectorXd::Zero(layer.biases.size());
    }

    // Clip gradients
    Eigen::MatrixXd clipped_dw = dw.cwiseMin(CLIP_VALUE).cwiseMax(-CLIP_VALUE);
    Eigen::VectorXd clipped_db = db.cwiseMin(CLIP_VALUE).cwiseMax(-CLIP_VALUE);

    // Update square gradients
    square_grad_w = decay_rate * square_grad_w + (1 - decay_rate) * clipped_dw.array().square().matrix();
    square_grad_b = decay_rate * square_grad_b + (1 - decay_rate) * clipped_db.array().square().matrix();

    // Clip and check for numerical instability
    clip_and_check(square_grad_w, "RMSprop square_grad_w");
    clip_and_check(square_grad_b, "RMSprop square_grad_b");

    // Compute updates with improved numerical stability
    Eigen::MatrixXd update_w = (learning_rate * clipped_dw.array() / (square_grad_w.array().sqrt() + epsilon).cwiseMax(epsilon)).matrix();
    Eigen::VectorXd update_b = (learning_rate * clipped_db.array() / (square_grad_b.array().sqrt() + epsilon).cwiseMax(epsilon)).matrix();

    // Clip and check updates
    clip_and_check(update_w, "RMSprop update_w");
    clip_and_check(update_b, "RMSprop update_b");

    // Apply updates
    layer.weights -= update_w;
    layer.biases -= update_b;

    if (layer.batch_norm)
    {
        Eigen::VectorXd clipped_gamma_grad = layer.bn_gamma_grad.cwiseMin(CLIP_VALUE).cwiseMax(-CLIP_VALUE);
        Eigen::VectorXd clipped_beta_grad = layer.bn_beta_grad.cwiseMin(CLIP_VALUE).cwiseMax(-CLIP_VALUE);

        // Initialize or resize square_grad_gamma and square_grad_beta if necessary
        if (square_grad_gamma.size() != clipped_gamma_grad.size())
        {
            square_grad_gamma = Eigen::VectorXd::Zero(clipped_gamma_grad.size());
        }
        if (square_grad_beta.size() != clipped_beta_grad.size())
        {
            square_grad_beta = Eigen::VectorXd::Zero(clipped_beta_grad.size());
        }

        // Update square gradients for gamma and beta
        square_grad_gamma = decay_rate * square_grad_gamma + (1 - decay_rate) * clipped_gamma_grad.array().square().matrix();
        square_grad_beta = decay_rate * square_grad_beta + (1 - decay_rate) * clipped_beta_grad.array().square().matrix();

        // Compute updates for gamma and beta
        Eigen::VectorXd update_gamma = (learning_rate * clipped_gamma_grad.array() / (square_grad_gamma.array().sqrt() + epsilon).cwiseMax(epsilon)).matrix();
        Eigen::VectorXd update_beta = (learning_rate * clipped_beta_grad.array() / (square_grad_beta.array().sqrt() + epsilon).cwiseMax(epsilon)).matrix();

        clip_and_check(update_gamma, "RMSprop update_gamma");
        clip_and_check(update_beta, "RMSprop update_beta");

        // Apply updates to gamma and beta
        layer.batch_norm->set_gamma(layer.batch_norm->get_gamma() - update_gamma);
        layer.batch_norm->set_beta(layer.batch_norm->get_beta() - update_beta);
    }
}

void GradientDescent::setLearningRate(double lr) { learning_rate = lr; }
double GradientDescent::getLearningRate() const { return learning_rate; }

void Adam::setLearningRate(double lr) { learning_rate = lr; }
double Adam::getLearningRate() const { return learning_rate; }
void Adam::setEpsilon(double eps) { epsilon = eps; }

void RMSprop::setLearningRate(double lr) { learning_rate = lr; }
double RMSprop::getLearningRate() const { return learning_rate; }
void RMSprop::setEpsilon(double eps) { epsilon = eps; }

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