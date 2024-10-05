#pragma once
#include <Eigen/Dense>
#include <memory>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "exceptions.hpp"
#include "optimization_algorithms.hpp"
#include "activation_functions.hpp"
#include "weight_initialization.hpp"
#include "regularization_types.hpp"
#include "logger.hpp"
#include <random>
#include <sstream>
#include <stdexcept>

class Optimizer;

class Layer
{
public:
    // Constructor
    Layer(int input_size, int output_size, ActivationType activation,
          std::unique_ptr<Optimizer> optimizer,
          double learning_rate,
          bool use_batch_norm = true,
          double momentum = 0.99,
          WeightInitialization weight_init = WeightInitialization::Xavier);

    // Layer connectivity methods
    void set_prev_layer(std::shared_ptr<Layer> layer);
    void set_next_layer(std::shared_ptr<Layer> layer);

    // Forward and backward propagation
    double feedforward(const Eigen::MatrixXd &input);
    void backpropagate(const Eigen::MatrixXd &output_gradient, double learning_rate);

    // Parallel computation methods (if implemented)
    void parallel_feedforward(const Eigen::MatrixXd &input);
    void parallel_backpropagate(const Eigen::MatrixXd &output_gradient, double learning_rate);

    // Regularization methods
    double compute_regularization_loss() const;
    void set_regularization(RegularizationType type, double strength);

    // Getters
    int get_input_size() const { return input_size; }
    int get_output_size() const { return output_size; }
    const Eigen::MatrixXd &get_last_output() const { return last_output; }

    // Setter for learning rate
    void set_learning_rate(double new_learning_rate);

private:
    // Layer parameters
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    Eigen::VectorXd gamma;
    Eigen::VectorXd beta;
    Eigen::VectorXd running_mean;
    Eigen::VectorXd running_variance;

    // Layer properties
    int input_size;
    int output_size;
    double momentum;
    bool use_batch_norm;
    ActivationType activation_type;
    std::unique_ptr<Optimizer> optimizer;
    RegularizationType regularization_type = RegularizationType::None;
    double regularization_strength = 0.0;
    double learning_rate;

    // Layer connections
    std::shared_ptr<Layer> prev_layer;
    std::shared_ptr<Layer> next_layer;

    // Caches for backpropagation
    Eigen::MatrixXd last_input;
    Eigen::MatrixXd last_output;
    Eigen::MatrixXd normalized_input;

    // Helper methods
    void clip_gradients(Eigen::MatrixXd &d_weights, Eigen::VectorXd &d_biases,
                        Eigen::VectorXd *d_gamma = nullptr, Eigen::VectorXd *d_beta = nullptr,
                        double clip_value = 1.0);
    Eigen::MatrixXd compute_regularization_gradient() const;

    // Use the standalone activation functions
    Eigen::MatrixXd activate(const Eigen::MatrixXd &x) const
    {
        return ::activate(x, activation_type);
    }

    Eigen::MatrixXd activate_derivative(const Eigen::MatrixXd &x) const
    {
        return ::activate_derivative(x, activation_type);
    }
};