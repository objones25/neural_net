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

    py::class_<NeuralNetwork> neural_network(m, "NeuralNetwork");

    py::enum_<NeuralNetwork::WeightInitialization>(neural_network, "WeightInitialization")
        .value("Random", NeuralNetwork::WeightInitialization::Random)
        .value("Xavier", NeuralNetwork::WeightInitialization::Xavier)
        .value("He", NeuralNetwork::WeightInitialization::He);

    py::enum_<NeuralNetwork::RegularizationType>(neural_network, "RegularizationType")
        .value("None", NeuralNetwork::RegularizationType::None)
        .value("L1", NeuralNetwork::RegularizationType::L1)
        .value("L2", NeuralNetwork::RegularizationType::L2);

    // Bind NeuralNetwork class
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
        .def("train", [](NeuralNetwork& self,
                         const std::vector<Eigen::VectorXd>& inputs,
                         const std::vector<Eigen::VectorXd>& targets,
                         int epochs,
                         int batch_size,
                         double error_tolerance) {
            try {
                self.train(inputs, targets, epochs, batch_size, error_tolerance);
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("Training failed: ") + e.what());
            }
        }, py::arg("inputs"), py::arg("targets"), py::arg("epochs"),
           py::arg("batch_size") = 32, py::arg("error_tolerance") = 1e-4)
        .def("predict", [](const NeuralNetwork& self, const Eigen::VectorXd& input) {
            try {
                return self.predict(input);
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("Prediction failed: ") + e.what());
            }
        })
        .def("get_loss", [](const NeuralNetwork& self,
                            const std::vector<Eigen::VectorXd>& inputs,
                            const std::vector<Eigen::VectorXd>& targets) {
            try {
                return self.get_loss(inputs, targets);
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("Loss calculation failed: ") + e.what());
            }
        });

    // Register custom exceptions
    py::register_exception<NetworkConfigurationError>(m, "NetworkConfigurationError");
    py::register_exception<TrainingDataError>(m, "TrainingDataError");

    // Bind optimization algorithms (if needed)
    py::class_<OptimizationAlgorithm, std::unique_ptr<OptimizationAlgorithm>>(m, "OptimizationAlgorithm");
    py::class_<GradientDescent, OptimizationAlgorithm>(m, "GradientDescent")
        .def(py::init<double>());
    py::class_<Adam, OptimizationAlgorithm>(m, "Adam")
        .def(py::init<double, double, double, double>());
    py::class_<RMSprop, OptimizationAlgorithm>(m, "RMSprop")
        .def(py::init<double, double, double>());

    m.def("create_optimizer", &create_optimizer);

    std::cout << "Module initialization complete" << std::endl;
}