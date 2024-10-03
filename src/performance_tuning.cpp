#ifdef _OPENMP
#include <omp.h>
#endif
#include "neural_network.hpp"
#include "layer.hpp"

void NeuralNetwork::enable_parallel_processing(int num_threads) {
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif
}

void Layer::parallel_feedforward(const Eigen::MatrixXd& input) {
    // Perform matrix multiplication
    Eigen::MatrixXd output = Eigen::MatrixXd::Zero(input.rows(), weights.cols());
    #pragma omp parallel for
    for (int i = 0; i < input.rows(); ++i) {
        output.row(i) = input.row(i) * weights;
        output.row(i) += biases.transpose();
    }
    
    // Apply activation function (this is already vectorized)
    last_output = activate(output);
}

void Layer::parallel_backpropagate(const Eigen::MatrixXd& output_gradient, double learning_rate) {
    Eigen::MatrixXd d_output = output_gradient.array() * activate_derivative(last_output).array();

    // Compute weight gradients
    Eigen::MatrixXd weight_gradients = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    
    #pragma omp parallel for
    for (int i = 0; i < last_input.rows(); ++i) {
        weight_gradients += last_input.row(i).transpose() * d_output.row(i);
    }

    // Update weights and biases
    #pragma omp parallel for
    for (int i = 0; i < weights.rows(); ++i) {
        for (int j = 0; j < weights.cols(); ++j) {
            weights(i, j) -= learning_rate * weight_gradients(i, j) / last_input.rows();
        }
    }

    biases -= learning_rate * d_output.colwise().mean();

    if (prev_layer) {
        Eigen::MatrixXd prev_layer_gradient = d_output * weights.transpose();
        prev_layer->parallel_backpropagate(prev_layer_gradient, learning_rate);
    }
}