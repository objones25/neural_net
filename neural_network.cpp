#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>

namespace py = pybind11;

class NeuralNetwork {
private:
    std::vector<int> layers;                                 // Stores the number of neurons in each layer
    std::vector<std::vector<std::vector<double>>> weights;   // 3D vector to store weights
    std::vector<std::vector<double>> biases;                 // 2D vector to store biases
    std::string activation_function;                         // Name of the activation function to use
    bool use_softmax_output;                                 // Flag to use softmax in the output layer

    // Mutable vectors to store intermediate results during forward pass
    mutable std::vector<std::vector<double>> activations;
    mutable std::vector<std::vector<double>> zs;
    
    // Activation function
    double activate(double x, bool is_output_layer) const;

    // Derivative of the activation function
    double activate_derivative(double x, bool is_output_layer) const;

    // Softmax activation function for the output layer
    std::vector<double> softmax(const std::vector<double>& x) const;

    // Backpropagation algorithm
    void backpropagate(const std::vector<double>& input, const std::vector<double>& target,
                       std::vector<std::vector<std::vector<double>>>& weight_gradients,
                       std::vector<std::vector<double>>& bias_gradients);

public:
    // Constructor
    NeuralNetwork(const std::vector<int>& layer_sizes, const std::string& activation = "sigmoid", bool use_softmax = false);

    // Forward pass: Compute the output for a given input
    std::vector<double> forward(const std::vector<double>& input) const;

    // Train the neural network using parallel batch stochastic gradient descent
    void train(const std::vector<std::vector<double>>& inputs, 
               const std::vector<std::vector<double>>& targets, 
               int epochs, double learning_rate, int batch_size, int num_threads);

    // Setter and getter methods
    void set_activation(const std::string& activation);
    [[nodiscard]] std::string get_activation() const;
    void set_softmax_output(bool use_softmax);
    [[nodiscard]] bool get_softmax_output() const;
};

// Function implementations

double NeuralNetwork::activate(double x, bool is_output_layer) const {
    if (is_output_layer && use_softmax_output) {
        return x; // For softmax, we return the input unchanged
    } else if (activation_function == "sigmoid") {
        return 1.0 / (1.0 + std::exp(-x));
    } else if (activation_function == "relu") {
        return std::max(0.0, x);
    } else if (activation_function == "tanh") {
        return std::tanh(x);
    } else {
        throw std::runtime_error("Unknown activation function");
    }
}

double NeuralNetwork::activate_derivative(double x, bool is_output_layer) const {
    if (is_output_layer && use_softmax_output) {
        return 1.0; // The derivative for softmax is handled differently in backpropagation
    } else if (activation_function == "sigmoid") {
        double sig = activate(x, false);
        return sig * (1 - sig);
    } else if (activation_function == "relu") {
        return x > 0 ? 1.0 : 0.0;
    } else if (activation_function == "tanh") {
        double tanh_x = std::tanh(x);
        return 1 - tanh_x * tanh_x;
    } else {
        throw std::runtime_error("Unknown activation function");
    }
}

std::vector<double> NeuralNetwork::softmax(const std::vector<double>& x) const {
    std::vector<double> result(x.size());
    double max_val = *std::max_element(x.begin(), x.end());
    double sum = 0.0;
    
    // Compute exp(x - max_val) for numerical stability
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i] - max_val);
        sum += result[i];
    }
    
    // Normalize
    for (double& val : result) {
        val /= sum;
    }
    
    return result;
}

void NeuralNetwork::backpropagate(const std::vector<double>& input, const std::vector<double>& target,
                   std::vector<std::vector<std::vector<double>>>& weight_gradients,
                   std::vector<std::vector<double>>& bias_gradients) {
    // Forward pass
    forward(input);
    
    // Backward pass
    std::vector<std::vector<double>> deltas;
    
    // Compute error for the output layer
    std::vector<double> output_error(target.size());
    if (use_softmax_output) {
        for (size_t i = 0; i < target.size(); ++i) {
            output_error[i] = activations.back()[i] - target[i];
        }
    } else {
        for (size_t i = 0; i < target.size(); ++i) {
            double error = activations.back()[i] - target[i];
            output_error[i] = error * activate_derivative(zs.back()[i], true);
        }
    }
    deltas.push_back(output_error);
    
    // Compute error for hidden layers
    for (int i = static_cast<int>(weights.size()) - 2; i >= 0; --i) {
        std::vector<double> layer_error(weights[i].size());
        for (size_t j = 0; j < weights[i].size(); ++j) {
            double error = 0.0;
            for (size_t k = 0; k < weights[i+1].size(); ++k) {
                error += weights[i+1][k][j] * deltas.front()[k];
            }
            layer_error[j] = error * activate_derivative(zs[i][j], false);
        }
        deltas.insert(deltas.begin(), layer_error);
    }
    
    // Accumulate gradients
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                weight_gradients[i][j][k] += deltas[i][j] * activations[i][k];
            }
            bias_gradients[i][j] += deltas[i][j];
        }
    }
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes, const std::string& activation, bool use_softmax) 
    : layers(layer_sizes), activation_function(activation), use_softmax_output(use_softmax) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    // Initialize weights and biases
    for (size_t i = 1; i < layers.size(); ++i) {
        weights.push_back(std::vector<std::vector<double>>(layers[i], std::vector<double>(layers[i-1])));
        biases.push_back(std::vector<double>(layers[i]));
        
        // Initialize weights with random values
        for (auto& neuron_weights : weights.back()) {
            for (auto& weight : neuron_weights) {
                weight = d(gen);
            }
        }
        
        // Initialize biases with random values
        for (auto& bias : biases.back()) {
            bias = d(gen);
        }
    }
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) const {
    activations.clear();
    zs.clear();
    activations.push_back(input);

    // Compute activations for each layer
    for (size_t i = 0; i < weights.size(); ++i) {
        std::vector<double> z(weights[i].size());
        std::vector<double> activation(weights[i].size());
        bool is_output_layer = (i == weights.size() - 1);
        
        for (size_t j = 0; j < weights[i].size(); ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                sum += weights[i][j][k] * activations.back()[k];
            }
            z[j] = sum + biases[i][j];
            activation[j] = activate(z[j], is_output_layer);
        }
        
        zs.push_back(z);
        activations.push_back(activation);

        // Apply softmax to the output layer if specified
        if (is_output_layer && use_softmax_output) {
            activations.back() = softmax(zs.back());
        }
    }
    
    return activations.back();
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs, 
           const std::vector<std::vector<double>>& targets, 
           int epochs, double learning_rate, int batch_size, int num_threads) {
    
    int num_samples = static_cast<int>(inputs.size());
    int num_batches = static_cast<int>(std::ceil(static_cast<double>(num_samples) / batch_size));

    // Create a vector of indices and shuffle it
    std::vector<int> indices(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        indices[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());

    // Mutex for thread-safe weight updates
    std::mutex weights_mutex;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle the indices at the start of each epoch
        std::shuffle(indices.begin(), indices.end(), g);

        // Process each batch
        for (int batch = 0; batch < num_batches; ++batch) {
            int start_idx = batch * batch_size;
            int end_idx = std::min((batch + 1) * batch_size, num_samples);

            // Accumulate gradients for the batch
            std::vector<std::vector<std::vector<double>>> weight_gradients(weights.size());
            std::vector<std::vector<double>> bias_gradients(biases.size());

            for (size_t i = 0; i < weights.size(); ++i) {
                weight_gradients[i].resize(weights[i].size());
                for (size_t j = 0; j < weights[i].size(); ++j) {
                    weight_gradients[i][j].resize(weights[i][j].size(), 0.0);
                }
                bias_gradients[i].resize(biases[i].size(), 0.0);
            }

            // Lambda function for processing a range of samples
            auto process_range = [&](int start, int end) {
                for (int i = start; i < end; ++i) {
                    int idx = indices[i];
                    backpropagate(inputs[idx], targets[idx], weight_gradients, bias_gradients);
                }
            };

            // Divide work among threads
            std::vector<std::thread> threads;
            int samples_per_thread = (end_idx - start_idx) / num_threads;
            for (int t = 0; t < num_threads; ++t) {
                int thread_start = start_idx + t * samples_per_thread;
                int thread_end = (t == num_threads - 1) ? end_idx : thread_start + samples_per_thread;
                threads.emplace_back(process_range, thread_start, thread_end);
            }

            // Wait for all threads to complete
            for (auto& thread : threads) {
                thread.join();
            }

            // Update weights and biases with accumulated gradients
            std::lock_guard<std::mutex> lock(weights_mutex);
            for (size_t i = 0; i < weights.size(); ++i) {
                for (size_t j = 0; j < weights[i].size(); ++j) {
                    for (size_t k = 0; k < weights[i][j].size(); ++k) {
                        weights[i][j][k] -= learning_rate * weight_gradients[i][j][k] / (end_idx - start_idx);
                    }
                    biases[i][j] -= learning_rate * bias_gradients[i][j] / (end_idx - start_idx);
                }
            }
        }
    }
}

void NeuralNetwork::set_activation(const std::string& activation) {
    if (activation != "sigmoid" && activation != "relu" && activation != "tanh") {
        throw std::runtime_error("Unsupported activation function");
    }
    activation_function = activation;
}

std::string NeuralNetwork::get_activation() const {
    return activation_function;
}

void NeuralNetwork::set_softmax_output(bool use_softmax) {
    use_softmax_output = use_softmax;
}

bool NeuralNetwork::get_softmax_output() const {
    return use_softmax_output;
}

// Python module definition
PYBIND11_MODULE(neural_network_cpp, m) {
    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<const std::vector<int>&, const std::string&, bool>(),
             py::arg("layer_sizes"), py::arg("activation") = "sigmoid", py::arg("use_softmax") = false)
        .def("forward", &NeuralNetwork::forward)
        .def("train", &NeuralNetwork::train,
             py::arg("inputs"), py::arg("targets"), py::arg("epochs"),
             py::arg("learning_rate"), py::arg("batch_size"), py::arg("num_threads"))
        .def("set_activation", &NeuralNetwork::set_activation)
        .def("get_activation", &NeuralNetwork::get_activation)
        .def("set_softmax_output", &NeuralNetwork::set_softmax_output)
        .def("get_softmax_output", &NeuralNetwork::get_softmax_output);
}