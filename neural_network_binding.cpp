#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "neural_network.hpp"
#include "optimization_algorithms.hpp"
#include "activation_functions.hpp"
#include "exceptions.hpp"
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(neural_network_py, m)
{
    std::cout << "Initializing module..." << std::endl;

    py::enum_<ActivationFunction::Type>(m, "ActivationType")
        .value("Linear", ActivationFunction::Type::Linear)
        .value("ReLU", ActivationFunction::Type::ReLU)
        .value("Sigmoid", ActivationFunction::Type::Sigmoid)
        .value("Tanh", ActivationFunction::Type::Tanh)
        .value("Softmax", ActivationFunction::Type::Softmax);

    std::cout << "ActivationType enum created" << std::endl;

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
        .def(py::init<const std::vector<int>&,
                      ActivationFunction::Type,
                      ActivationFunction::Type,
                      NeuralNetwork::WeightInitialization,
                      const std::string&,
                      double,
                      NeuralNetwork::RegularizationType,
                      double>(),
             py::arg("layer_sizes"),
             py::arg("hidden_activation") = ActivationFunction::Type::ReLU,
             py::arg("output_activation") = ActivationFunction::Type::Sigmoid,
             py::arg("weight_init") = NeuralNetwork::WeightInitialization::Random,
             py::arg("optimizer_name") = "GradientDescent",
             py::arg("learning_rate") = 0.01,
             py::arg("reg_type") = NeuralNetwork::RegularizationType::None,
             py::arg("reg_strength") = 0.0)
        .def("train", &NeuralNetwork::train,
             py::arg("inputs"), py::arg("targets"), py::arg("epochs"),
             py::arg("batch_size") = 32, py::arg("error_tolerance") = 1e-4,
             py::arg("validation_split") = 0.2)
        .def("predict", &NeuralNetwork::predict)
        .def("get_loss", &NeuralNetwork::get_loss);

    std::cout << "NeuralNetwork methods bound" << std::endl;

    // Add custom exceptions
    py::register_exception<NetworkConfigurationError>(m, "NetworkConfigurationError");
    py::register_exception<TrainingDataError>(m, "TrainingDataError");

    std::cout << "Custom exceptions registered" << std::endl;

    // Bind optimization algorithms
    py::class_<OptimizationAlgorithm, std::unique_ptr<OptimizationAlgorithm>>(m, "OptimizationAlgorithm");
    py::class_<GradientDescent, OptimizationAlgorithm>(m, "GradientDescent")
        .def(py::init<double>());
    py::class_<Adam, OptimizationAlgorithm>(m, "Adam")
        .def(py::init<double, double, double, double>());
    py::class_<RMSprop, OptimizationAlgorithm>(m, "RMSprop")
        .def(py::init<double, double, double>());

    m.def("create_optimizer", &create_optimizer);

    std::cout << "Optimization algorithms bound" << std::endl;

    std::cout << "Module initialization complete" << std::endl;
}