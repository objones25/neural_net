#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "activation_functions.hpp"
#include "optimization_algorithms.hpp"
#include "batch_normalization.hpp"

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
        None,
        L1,
        L2
    };
    const std::vector<int> &getLayers() const { return layers; }
    const std::vector<Eigen::MatrixXd> &getWeights() const { return weights; }

    NeuralNetwork(const std::vector<int> &layer_sizes,
                  ActivationFunction::Type hidden_activation = ActivationFunction::Type::ReLU,
                  ActivationFunction::Type output_activation = ActivationFunction::Type::Sigmoid,
                  WeightInitialization weight_init = WeightInitialization::Random,
                  const std::string &optimizer_name = "GradientDescent",
                  double learning_rate = 0.01,
                  RegularizationType reg_type = RegularizationType::None,
                  double reg_strength = 0.0);

    void train(const std::vector<Eigen::VectorXd> &inputs,
               const std::vector<Eigen::VectorXd> &targets,
               int epochs,
               int batch_size,
               double error_tolerance = 1e-4);

    Eigen::VectorXd predict(const Eigen::VectorXd &input) const;
    double get_loss(const std::vector<Eigen::VectorXd>& inputs,
                    const std::vector<Eigen::VectorXd>& targets) const;

    // Add this public static method
    static std::unique_ptr<OptimizationAlgorithm> create_optimizer_for_network(const std::string &name, double learning_rate);
    void reset();

private:
    std::vector<int> layers;
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases;
    std::vector<BatchNorm> batch_norms;
    ActivationFunction activation_function;
    WeightInitialization weight_init;
    std::unique_ptr<OptimizationAlgorithm> optimizer;
    RegularizationType regularization_type;
    double regularization_strength;

    void initialize_weights();
    void validate() const;
    Eigen::VectorXd feedforward(const Eigen::VectorXd &input) const;
    std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
    feedforward_with_intermediates(const Eigen::VectorXd &input) const;
    std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>>
    backpropagate(const Eigen::VectorXd &input, const Eigen::VectorXd &target);
    void apply_regularization(std::vector<Eigen::MatrixXd> &weight_gradients,
                              std::vector<Eigen::VectorXd> &bias_gradients);
    void update_batch(const std::vector<Eigen::VectorXd> &batch_inputs,
                      const std::vector<Eigen::VectorXd> &batch_targets);
    void check_input_size(const Eigen::VectorXd &input) const;
    void check_target_size(const Eigen::VectorXd &target) const;
    bool is_valid(const Eigen::MatrixXd &mat) const;
    bool is_valid(const Eigen::VectorXd &vec) const;
    void check_gradients(const Eigen::VectorXd &input, const Eigen::VectorXd &target);
};