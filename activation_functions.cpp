#include "activation_functions.hpp"
#include <stdexcept>

// Constructor for ActivationFunction class
ActivationFunction::ActivationFunction(Type hiddenType, Type outputType)
    : hiddenActivation(getFunction(hiddenType)),
      hiddenDerivative(getDerivative(hiddenType)),
      outputActivation(getFunction(outputType)),
      outputDerivative(getDerivative(outputType)),
      outputType(outputType) {}

// Apply hidden layer activation function
Eigen::VectorXd ActivationFunction::activateHidden(const Eigen::VectorXd &x) const { return hiddenActivation(x); }

// Apply hidden layer activation function derivative
Eigen::VectorXd ActivationFunction::derivativeHidden(const Eigen::VectorXd &x) const { return hiddenDerivative(x); }

// Apply output layer activation function
Eigen::VectorXd ActivationFunction::activateOutput(const Eigen::VectorXd &x) const { return outputActivation(x); }

// Apply output layer activation function derivative
Eigen::VectorXd ActivationFunction::derivativeOutput(const Eigen::VectorXd &x) const { return outputDerivative(x); }

// Get the activation function based on the specified type
std::function<Eigen::VectorXd(const Eigen::VectorXd &)> ActivationFunction::getFunction(Type type)
{
    switch (type)
    {
    case Type::Linear:
        return [](const Eigen::VectorXd &x) { return x; };
    case Type::ReLU:
        return [](const Eigen::VectorXd &x) { return x.array().max(0.0); };
    case Type::Sigmoid:
        return [](const Eigen::VectorXd &x) { return 1.0 / (1.0 + (-x.array().min(50.0).max(-50.0)).exp()); };
    case Type::Tanh:
        return [](const Eigen::VectorXd &x) { return x.array().tanh(); };
    case Type::Softmax:
        return [](const Eigen::VectorXd &x)
        {
            Eigen::VectorXd exp_x = (x.array() - x.maxCoeff()).exp();
            return exp_x.array() / (exp_x.sum() + 1e-10);
        };
    default:
        throw std::invalid_argument("Unknown activation function type");
    }
}

// Get the derivative of the activation function based on the specified type
std::function<Eigen::VectorXd(const Eigen::VectorXd &)> ActivationFunction::getDerivative(Type type)
{
    switch (type)
    {
    case Type::Linear:
        return [](const Eigen::VectorXd &x) { return Eigen::VectorXd::Ones(x.size()); };
    case Type::ReLU:
        return [](const Eigen::VectorXd &x) { return (x.array() > 0.0).cast<double>(); };
    case Type::Sigmoid:
        return [](const Eigen::VectorXd &x)
        {
            Eigen::VectorXd s = 1.0 / (1.0 + (-x.array()).exp());
            return s.array() * (1.0 - s.array());
        };
    case Type::Tanh:
        return [](const Eigen::VectorXd &x)
        {
            Eigen::VectorXd t = x.array().tanh();
            return 1.0 - t.array().square();
        };
    case Type::Softmax:
        return [](const Eigen::VectorXd &x)
        {
            Eigen::VectorXd s = x.array().exp();
            s /= s.sum();
            return s.array() * (1.0 - s.array());
        };
    default:
        throw std::invalid_argument("Unknown activation function type");
    }
}