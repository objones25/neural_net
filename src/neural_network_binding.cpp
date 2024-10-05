#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "neural_network.hpp"
#include "activation_functions.hpp"
#include "weight_initialization.hpp"
#include <iostream>

namespace py = pybind11;

// Standalone test_enums function
void test_enums()
{
    std::cout << "ReLU: " << static_cast<int>(ActivationType::ReLU) << std::endl;
    std::cout << "Xavier: " << static_cast<int>(WeightInitialization::Xavier) << std::endl;
    std::cout << "MeanSquaredError: " << static_cast<int>(LossFunction::MeanSquaredError) << std::endl;
}

// Helper function to get enum values
template <typename EnumType>
py::list enum_values()
{
    py::list values;
    for (int i = 0; i < static_cast<int>(EnumType::_COUNT); ++i)
    {
        values.append(static_cast<EnumType>(i));
    }
    return values;
}

PYBIND11_MODULE(neural_network_py, m)
{
    py::enum_<ActivationType>(m, "ActivationType", py::arithmetic())
        .value("Linear", ActivationType::Linear)
        .value("ReLU", ActivationType::ReLU)
        .value("Sigmoid", ActivationType::Sigmoid)
        .value("Tanh", ActivationType::Tanh)
        .value("Softmax", ActivationType::Softmax)
        .export_values();
    m.def("ActivationType_values", &enum_values<ActivationType>);

    py::enum_<WeightInitialization>(m, "WeightInitialization", py::arithmetic())
        .value("Xavier", WeightInitialization::Xavier)
        .value("He", WeightInitialization::He)
        .value("LeCun", WeightInitialization::LeCun)
        .export_values();
    m.def("WeightInitialization_values", &enum_values<WeightInitialization>);

    py::enum_<LossFunction>(m, "LossFunction", py::arithmetic())
        .value("MeanSquaredError", LossFunction::MeanSquaredError)
        .value("CrossEntropy", LossFunction::CrossEntropy)
        .export_values();
    m.def("LossFunction_values", &enum_values<LossFunction>);

    m.def("test_enums", &test_enums, "Test function for enums");

    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<const std::vector<int> &, ActivationType, ActivationType, const std::string &, double, LossFunction, bool, WeightInitialization>())
        .def("predict", &NeuralNetwork::predict_matrix)
        .def("predict", [](NeuralNetwork &self, const Eigen::MatrixXd &input)
             { return self.predict(input); })
        .def("get_input_size", &NeuralNetwork::get_input_size)
        .def("train", &NeuralNetwork::train)
        .def("calculate_loss", &NeuralNetwork::calculate_loss)
        .def("set_learning_rate", &NeuralNetwork::set_learning_rate)
        .def("get_learning_rate", &NeuralNetwork::get_learning_rate)
        .def("get_activation_types", &NeuralNetwork::get_activation_types)
        .def_static("enable_debug_logging", &NeuralNetwork::enable_debug_logging);

    // Add custom exceptions
    py::register_exception<NetworkConfigurationError>(m, "NetworkConfigurationError");
    py::register_exception<TrainingDataError>(m, "TrainingDataError");
    py::register_exception<NumericalInstabilityError>(m, "NumericalInstabilityError");
    py::register_exception<SizeMismatchError>(m, "SizeMismatchError");
}