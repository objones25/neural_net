#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "main.hpp"
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(neural_network_py, m)
{
    std::cout << "Initializing module..." << std::endl;

    py::enum_<ActivationFunction>(m, "ActivationFunction")
        .value("Sigmoid", ActivationFunction::Sigmoid)
        .value("ReLU", ActivationFunction::ReLU)
        .value("TanH", ActivationFunction::TanH);

    std::cout << "ActivationFunction enum created" << std::endl;

    py::enum_<OptimizationAlgorithm>(m, "OptimizationAlgorithm")
        .value("GradientDescent", OptimizationAlgorithm::GradientDescent)
        .value("Adam", OptimizationAlgorithm::Adam)
        .value("RMSprop", OptimizationAlgorithm::RMSprop);

    std::cout << "OptimizationAlgorithm enum created" << std::endl;

    py::class_<NeuralNetwork> neural_network(m, "NeuralNetwork");

    std::cout << "NeuralNetwork class created" << std::endl;

    py::enum_<NeuralNetwork::WeightInitialization>(neural_network, "WeightInitialization")
        .value("Random", NeuralNetwork::WeightInitialization::Random)
        .value("Xavier", NeuralNetwork::WeightInitialization::Xavier)
        .value("He", NeuralNetwork::WeightInitialization::He);

    std::cout << "WeightInitialization enum created" << std::endl;

    py::enum_<NeuralNetwork::RegularizationType>(neural_network, "RegularizationType")
        .value("None", NeuralNetwork::RegularizationType::None)
        .value("L1", NeuralNetwork::RegularizationType::L1)
        .value("L2", NeuralNetwork::RegularizationType::L2);

    std::cout << "RegularizationType enum created" << std::endl;

    neural_network
        .def(py::init<const std::vector<int> &, double, ActivationFunction, NeuralNetwork::WeightInitialization, OptimizationAlgorithm, NeuralNetwork::RegularizationType, double>(),
             py::arg("layer_sizes"),
             py::arg("lr") = 0.01,
             py::arg("act_func") = ActivationFunction::Sigmoid,
             py::arg("weight_init") = NeuralNetwork::WeightInitialization::Random,
             py::arg("opt_algo") = OptimizationAlgorithm::GradientDescent,
             py::arg("reg_type") = NeuralNetwork::RegularizationType::None,
             py::arg("reg_strength") = 0.0)
        .def("train", &NeuralNetwork::train,
             py::arg("inputs"), py::arg("targets"), py::arg("epochs"),
             py::arg("batch_size") = 32, py::arg("error_tolerance") = 1e-4)
        .def("predict", &NeuralNetwork::predict)
        .def("get_loss", &NeuralNetwork::get_loss)
        .def("save_weights", &NeuralNetwork::save_weights)
        .def("load_weights", &NeuralNetwork::load_weights);

    std::cout << "NeuralNetwork methods bound" << std::endl;

    // Add custom exceptions
    py::register_exception<NetworkConfigurationError>(m, "NetworkConfigurationError");
    py::register_exception<TrainingDataError>(m, "TrainingDataError");

    std::cout << "Custom exceptions registered" << std::endl;

    // Expose activation functions
    m.def("activate", &activate, "Apply activation function to input");
    m.def("activate_derivative", &activate_derivative, "Compute derivative of activation function");

    std::cout << "Activation functions exposed" << std::endl;

    std::cout << "Module initialization complete" << std::endl;
}