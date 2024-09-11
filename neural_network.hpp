#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>
#include "activation_functions.hpp"
#include "optimization_algorithms.hpp"

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

    NeuralNetwork(const std::vector<int> &layer_sizes,
                  double lr = 0.01,
                  ActivationFunction act_func = ActivationFunction::Sigmoid,
                  WeightInitialization weight_init = WeightInitialization::Random,
                  OptimizationAlgorithm opt_algo = OptimizationAlgorithm::GradientDescent,
                  RegularizationType reg_type = RegularizationType::None,
                  double reg_strength = 0.0);

    void validate() const;
    std::vector<Eigen::VectorXd> feedforward(const Eigen::VectorXd &input) const;
    void backpropagate(const Eigen::VectorXd &input, const Eigen::VectorXd &target);
    void train(const std::vector<Eigen::VectorXd>& inputs,
           const std::vector<Eigen::VectorXd>& targets,
           int epochs,
           int batch_size,
           double error_tolerance,
           double validation_split);
    std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>> compute_gradients(const Eigen::VectorXd &input, const Eigen::VectorXd &target);
    void apply_regularization(std::vector<Eigen::MatrixXd> &weight_gradients,
                              std::vector<Eigen::VectorXd> &bias_gradients);
    Eigen::VectorXd predict(const Eigen::VectorXd &input) const;
    double get_loss(const std::vector<Eigen::VectorXd> &inputs,
                    const std::vector<Eigen::VectorXd> &targets) const;
    void save_weights(const std::string &filename) const;
    void load_weights(const std::string &filename);

private:
    std::vector<int> layers;
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases;
    double learning_rate;
    ActivationFunction activation_function;
    WeightInitialization weight_init;
    OptimizationAlgorithm optimization_algo;
    RegularizationType regularization_type;
    double regularization_strength;

    std::vector<Eigen::MatrixXd> m_weights;
    std::vector<Eigen::VectorXd> m_biases;
    std::vector<Eigen::MatrixXd> v_weights;
    std::vector<Eigen::VectorXd> v_biases;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    int t = 0;

    void initialize_weights();
    void check_input_size(const Eigen::VectorXd &input) const;
    void check_target_size(const Eigen::VectorXd &target) const;
};