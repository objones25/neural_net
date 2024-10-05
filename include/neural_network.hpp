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
    CrossEntropy,
    _COUNT
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
        Logger::log("Input shape: (" + std::to_string(input.rows()) + ", " + std::to_string(input.cols()) + ")");

        typename Derived::PlainObject current_input = input;
        for (size_t i = 0; i < layers.size(); ++i)
        {
            Logger::log("Processing layer " + std::to_string(i));
            Logger::log("Layer " + std::to_string(i) + " input shape: (" +
                        std::to_string(current_input.rows()) + ", " +
                        std::to_string(current_input.cols()) + ")");

            try
            {
                layers[i]->feedforward(current_input);
                current_input = layers[i]->get_last_output();
                Logger::log("Layer " + std::to_string(i) + " output shape: (" +
                            std::to_string(current_input.rows()) + ", " +
                            std::to_string(current_input.cols()) + ")");
            }
            catch (const std::exception &e)
            {
                Logger::log("Error in layer " + std::to_string(i) + ": " + e.what());
                throw;
            }
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
               int epochs, int batch_size, double learning_rate,
               double validation_split = 0.2, int patience = 10, double min_delta = 1e-4);
    double calculate_loss(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y);
    void set_learning_rate(double new_learning_rate);
    double get_learning_rate() const;
    void enable_parallel_processing(int num_threads);
    int get_input_size() const { return layers.front()->get_input_size(); }
    static void enable_debug_logging(bool enable)
    {
        Logger::set_debug_mode(enable);
    }

private:
    std::vector<std::shared_ptr<Layer>> layers;
    double learning_rate;
    LossFunction loss_function;
    ActivationType hidden_activation;
    ActivationType output_activation;
};