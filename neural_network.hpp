#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "activation_functions.hpp"
#include "optimization_algorithms.hpp"
#include "layer.hpp"

class NeuralNetwork
{
public:
    enum class WeightInitialization
    {
        Random,
        Xavier,
        He
    };
    enum class RegularizationType
    {
        NONE,
        L1,
        L2
    };

    void set_weights(const std::vector<Eigen::MatrixXd>& new_weights);
    void set_biases(const std::vector<Eigen::VectorXd>& new_biases);
    const std::vector<Layer>& getLayers() const { return layers; }
    void set_debug(bool debug) { debug_mode = debug; }
    bool get_debug() const { return debug_mode; }

    NeuralNetwork(const std::vector<int> &layer_sizes,
                  ActivationFunction::Type hidden_activation = ActivationFunction::Type::ReLU,
                  ActivationFunction::Type output_activation = ActivationFunction::Type::Sigmoid,
                  WeightInitialization weight_init = WeightInitialization::Random,
                  const std::string &optimizer_name = "GradientDescent",
                  double learning_rate = 0.01,
                  RegularizationType reg_type = RegularizationType::NONE,
                  double reg_strength = 0.0,
                  double learning_rate_adjustment = 1.0,
                  bool use_batch_norm = true);

    void train(const std::vector<Eigen::VectorXd> &inputs,
               const std::vector<Eigen::VectorXd> &targets,
               int epochs,
               int batch_size,
               double error_tolerance = 1e-4);

    Eigen::VectorXd predict(const Eigen::VectorXd &input) const;
    double get_loss(const std::vector<Eigen::VectorXd> &inputs,
                    const std::vector<Eigen::VectorXd> &targets) const;

    static std::unique_ptr<OptimizationAlgorithm> create_optimizer_for_network(const std::string &name, double learning_rate);
    void reset();
    void check_gradients(const Eigen::VectorXd &input, const Eigen::VectorXd &target);

private:
    std::vector<Layer> layers;
    std::vector<int> layer_sizes;
    ActivationFunction activation_function;
    WeightInitialization weight_init;
    std::unique_ptr<OptimizationAlgorithm> optimizer;
    RegularizationType regularization_type;
    double regularization_strength;
    bool use_batch_norm;
    bool debug_mode = false;

    void initialize_weights();
    void validate() const;
    Eigen::VectorXd feedforward(const Eigen::VectorXd &input) const;
    std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
    feedforward_with_intermediates(const Eigen::VectorXd &input) const;
    std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>>
    backpropagate(const Eigen::Ref<const Eigen::VectorXd> &input,
                  const Eigen::Ref<const Eigen::VectorXd> &target);
    void apply_regularization(std::vector<Eigen::MatrixXd> &weight_gradients,
                              std::vector<Eigen::VectorXd> &bias_gradients);
    void check_weights_initialization() const;
    void update_batch(const std::vector<Eigen::VectorXd> &batch_inputs,
                      const std::vector<Eigen::VectorXd> &batch_targets);
    void check_input_size(const Eigen::VectorXd &input) const;
    void check_target_size(const Eigen::VectorXd &target) const;
    bool is_valid(const Eigen::MatrixXd &mat) const;
    bool is_valid(const Eigen::VectorXd &vec) const;
    void debug_print(const std::string& message) const {
        if (debug_mode) {
            std::cout << "[DEBUG] " << message << std::endl;
        }
    }
};