#pragma once
#include <vector>
#include <memory>
#include <Eigen/Dense>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "layer.hpp"

enum class LossFunction
{
    MeanSquaredError,
    CrossEntropy
};

class NeuralNetwork
{
public:
    NeuralNetwork(const std::vector<int> &layer_sizes,
                  ActivationType hidden_activation,
                  ActivationType output_activation,
                  const std::string &optimizer_name,
                  double learning_rate,
                  LossFunction loss_function = LossFunction::MeanSquaredError,
                  bool use_batch_norm = true,
                  WeightInitialization weight_init = WeightInitialization::Xavier);

    template <typename Derived>
    typename Derived::PlainObject predict(const Eigen::MatrixBase<Derived> &input) const
    {
        Logger::log("Predict function called");
        Logger::log("Input size: " + std::to_string(input.cols()));
        Logger::log("Expected input size: " + std::to_string(layers.front()->get_input_size()));

        if (input.cols() != layers.front()->get_input_size())
        {
            throw SizeMismatchError("Input size does not match network input layer size");
        }

        typename Derived::PlainObject current_input = input;
        for (const auto &layer : layers)
        {
            layer->feedforward(current_input);
            current_input = layer->get_last_output();
        }
        return current_input;
    }

    // Add this non-template version for pybind11 binding
    Eigen::MatrixXd predict_matrix(const Eigen::MatrixXd &input) const
    {
        Logger::log("predict_matrix function called");
        return predict(input);
    }

    std::pair<ActivationType, ActivationType> get_activation_types() const
    {
        return {hidden_activation, output_activation};
    }

    void train(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y,
               int epochs, int batch_size, double learning_rate);
    double calculate_loss(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y);
    void set_learning_rate(double new_learning_rate);
    double get_learning_rate() const;
    void enable_parallel_processing(int num_threads);
    int get_input_size() const { return layers.front()->get_input_size(); }

private:
    std::vector<std::shared_ptr<Layer>> layers;
    double learning_rate;
    LossFunction loss_function;
    ActivationType hidden_activation;
    ActivationType output_activation;
};