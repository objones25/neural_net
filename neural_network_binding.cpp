#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "neural_network.hpp"
#include "optimization_algorithms.hpp"
#include "activation_functions.hpp"
#include "exceptions.hpp"
#include <iostream>
#include <stdexcept>

namespace py = pybind11;

PYBIND11_MODULE(neural_network_py, m)
{
    std::cout << "Initializing module..." << std::endl;

    // Bind enums
    py::enum_<ActivationFunction::Type>(m, "ActivationType")
        .value("Linear", ActivationFunction::Type::Linear)
        .value("ReLU", ActivationFunction::Type::ReLU)
        .value("Sigmoid", ActivationFunction::Type::Sigmoid)
        .value("Tanh", ActivationFunction::Type::Tanh)
        .value("Softmax", ActivationFunction::Type::Softmax);

    // Define NeuralNetwork class and its nested enums
    py::class_<NeuralNetwork> neural_network(m, "NeuralNetwork");

    py::enum_<NeuralNetwork::WeightInitialization>(neural_network, "WeightInitialization")
        .value("Random", NeuralNetwork::WeightInitialization::Random)
        .value("Xavier", NeuralNetwork::WeightInitialization::Xavier)
        .value("He", NeuralNetwork::WeightInitialization::He);

    py::enum_<NeuralNetwork::RegularizationType>(neural_network, "RegularizationType")
        .value("NONE", NeuralNetwork::RegularizationType::NONE)
        .value("L1", NeuralNetwork::RegularizationType::L1)
        .value("L2", NeuralNetwork::RegularizationType::L2);

    // Bind BatchNorm class
    py::class_<BatchNorm>(m, "BatchNorm")
        .def(py::init<int, double, double>())
        .def("forward", &BatchNorm::forward)
        .def("backward", &BatchNorm::backward)
        .def("get_gamma", &BatchNorm::get_gamma)
        .def("get_beta", &BatchNorm::get_beta)
        .def("set_gamma", &BatchNorm::set_gamma)
        .def("set_beta", &BatchNorm::set_beta);

    // Bind Layer class (single definition)
    py::class_<Layer>(m, "Layer")
        .def(py::init<int, int, bool>())
        .def_readwrite("weights", &Layer::weights)
        .def_readwrite("biases", &Layer::biases)
        .def_readwrite("bn_gamma_grad", &Layer::bn_gamma_grad)
        .def_readwrite("bn_beta_grad", &Layer::bn_beta_grad)
        .def_property("batch_norm", 
            [](const Layer &l) -> BatchNorm * { return l.batch_norm.get(); },
            [](Layer &l, BatchNorm *bn) { 
                if (bn == nullptr) {
                    throw std::invalid_argument("Cannot set batch_norm to null");
                }
                l.batch_norm.reset(bn); 
            })
        .def("create_batch_norm", [](Layer &l, int features, double momentum, double epsilon)
             { l.batch_norm = std::make_unique<BatchNorm>(features, momentum, epsilon); });

    // Bind NeuralNetwork class methods and constructor
    neural_network
        .def(py::init<const std::vector<int> &,
                      ActivationFunction::Type,
                      ActivationFunction::Type,
                      NeuralNetwork::WeightInitialization,
                      const std::string &,
                      double,
                      NeuralNetwork::RegularizationType,
                      double,
                      double,
                      bool>(),
             py::arg("layer_sizes"),
             py::arg("hidden_activation") = ActivationFunction::Type::ReLU,
             py::arg("output_activation") = ActivationFunction::Type::Sigmoid,
             py::arg("weight_init") = NeuralNetwork::WeightInitialization::Random,
             py::arg("optimizer_name") = "GradientDescent",
             py::arg("learning_rate") = 0.01,
             py::arg("reg_type") = NeuralNetwork::RegularizationType::NONE,
             py::arg("reg_strength") = 0.0,
             py::arg("learning_rate_adjustment") = 1.0,
             py::arg("use_batch_norm") = true)
        .def("train", &NeuralNetwork::train,
             py::arg("inputs"), py::arg("targets"), py::arg("epochs"), py::arg("batch_size") = 32, py::arg("error_tolerance") = 1e-4)
        .def("predict", &NeuralNetwork::predict)
        .def("get_loss", &NeuralNetwork::get_loss)
        .def("set_weights", &NeuralNetwork::set_weights)
        .def("set_biases", &NeuralNetwork::set_biases)
        .def("getLayers", &NeuralNetwork::getLayers, py::return_value_policy::reference_internal)
        .def("check_gradients", &NeuralNetwork::check_gradients, py::arg("input"), py::arg("target"))
        .def("set_debug", &NeuralNetwork::set_debug)
        .def("get_debug", &NeuralNetwork::get_debug)
        .def("set_learning_rate", &NeuralNetwork::set_learning_rate)
        .def("get_learning_rate", &NeuralNetwork::get_learning_rate);

    // Register custom exceptions
    py::register_exception<NetworkConfigurationError>(m, "NetworkConfigurationError");
    py::register_exception<TrainingDataError>(m, "TrainingDataError");
    py::register_exception<BatchNormalizationError>(m, "BatchNormalizationError");
    py::register_exception<OptimizerError>(m, "OptimizerError");
    py::register_exception<NumericalInstabilityError>(m, "NumericalInstabilityError");

    // Bind optimization algorithms
    py::class_<OptimizationAlgorithm, std::unique_ptr<OptimizationAlgorithm>>(m, "OptimizationAlgorithm");
    py::class_<GradientDescent, OptimizationAlgorithm>(m, "GradientDescent")
        .def(py::init<double>());
    py::class_<Adam, OptimizationAlgorithm>(m, "Adam")
        .def(py::init<double, double, double, double>());
    py::class_<RMSprop, OptimizationAlgorithm>(m, "RMSprop")
        .def(py::init<double, double, double>());

    std::cout << "Module initialization complete" << std::endl;
}