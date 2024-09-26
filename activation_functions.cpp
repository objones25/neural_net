#include "activation_functions.hpp"
#include <stdexcept>
#include <iostream>
#include <limits>

// Constructor for ActivationFunction class
ActivationFunction::ActivationFunction(Type hiddenType, Type outputType)
    : hiddenActivation(getFunction(hiddenType)),
      hiddenDerivative(getDerivative(hiddenType)),
      outputActivation(getFunction(outputType)),
      outputDerivative(getDerivative(outputType)),
      outputType(outputType)
{
    if (!hiddenActivation || !hiddenDerivative || !outputActivation || !outputDerivative)
    {
        throw std::runtime_error("Failed to initialize activation functions");
    }
}

// Apply hidden layer activation function
Eigen::VectorXd ActivationFunction::activateHidden(const Eigen::VectorXd &x) const
{
    if (x.size() == 0)
    {
        throw std::invalid_argument("Input vector is empty in activateHidden");
    }
    //std::cout << "Calculating hidden activation, input size: " << x.size() << std::endl;
    //std::cout << "Hidden activation input: " << x.transpose() << std::endl;
    Eigen::VectorXd result = hiddenActivation(x);
    //std::cout << "Hidden activation result: " << result.transpose() << std::endl;
    //std::cout << "Hidden activation calculation complete, output size: " << result.size() << std::endl;
    return result;
}

// Apply hidden layer activation function derivative
Eigen::VectorXd ActivationFunction::derivativeHidden(const Eigen::VectorXd &x) const
{
    if (x.size() == 0)
    {
        throw std::invalid_argument("Input vector is empty in derivativeHidden");
    }
    //std::cout << "Calculating hidden derivative, input size: " << x.size() << std::endl;
    Eigen::VectorXd result = hiddenDerivative(x);
    //std::cout << "Hidden derivative calculation complete, output size: " << result.size() << std::endl;
    return result;
}

// Apply output layer activation function
Eigen::VectorXd ActivationFunction::activateOutput(const Eigen::VectorXd &x) const
{
    if (x.size() == 0)
    {
        throw std::invalid_argument("Input vector is empty in activateOutput");
    }
    //std::cout << "Calculating output activation, input size: " << x.size() << std::endl;
    Eigen::VectorXd result = outputActivation(x);
    //std::cout << "Output activation calculation complete, output size: " << result.size() << std::endl;
    return result;
}

// Apply output layer activation function derivative
Eigen::VectorXd ActivationFunction::derivativeOutput(const Eigen::VectorXd &x) const
{
    if (x.size() == 0)
    {
        throw std::invalid_argument("Input vector is empty in derivativeOutput");
    }
    //std::cout << "Calculating output derivative, input size: " << x.size() << std::endl;
    Eigen::VectorXd result = outputDerivative(x);
    //std::cout << "Output derivative calculation complete, output size: " << result.size() << std::endl;
    return result;
}

// Get the activation function based on the specified type
std::function<Eigen::VectorXd(const Eigen::VectorXd &)> ActivationFunction::getFunction(Type type)
{
    if (type < Type::Linear || type > Type::Softmax)
    {
        throw std::invalid_argument("Invalid activation function type");
    }

    switch (type)
    {
    case Type::Linear:
        return [](const Eigen::VectorXd &x) noexcept -> Eigen::VectorXd
        {
            try
            {
                return x;
            }
            catch (...)
            {
                std::cerr << "Error in Linear activation" << std::endl;
                return Eigen::VectorXd::Zero(x.size());
            }
        };
    case Type::ReLU:
        return [](const Eigen::VectorXd &x) noexcept -> Eigen::VectorXd
        {
            try
            {
                //std::cout << "Applying ReLU activation" << std::endl;
                Eigen::VectorXd result = x.cwiseMax(0.0);
                //std::cout << "ReLU input: " << x.transpose() << std::endl;
                //std::cout << "ReLU output: " << result.transpose() << std::endl;
                return result;
            }
            catch (...)
            {
                std::cerr << "Error in ReLU activation" << std::endl;
                return Eigen::VectorXd::Zero(x.size());
            }
        };
    case Type::Sigmoid:
        return [](const Eigen::VectorXd &x) noexcept -> Eigen::VectorXd
        {
            try
            {
                return 1.0 / (1.0 + (-x.array().min(50.0).max(-50.0)).exp());
            }
            catch (...)
            {
                std::cerr << "Error in Sigmoid activation" << std::endl;
                return Eigen::VectorXd::Zero(x.size());
            }
        };
    case Type::Tanh:
        return [](const Eigen::VectorXd &x) noexcept -> Eigen::VectorXd
        {
            try
            {
                return x.array().tanh();
            }
            catch (...)
            {
                std::cerr << "Error in Tanh activation" << std::endl;
                return Eigen::VectorXd::Zero(x.size());
            }
        };
    case Type::Softmax:
        return [](const Eigen::VectorXd &x) noexcept -> Eigen::VectorXd
        {
            try
            {
                Eigen::VectorXd exp_x = (x.array() - x.maxCoeff()).exp();
                return exp_x.array() / (exp_x.sum() + std::numeric_limits<double>::epsilon());
            }
            catch (...)
            {
                std::cerr << "Error in Softmax activation" << std::endl;
                return Eigen::VectorXd::Zero(x.size());
            }
        };
    default:
        throw std::invalid_argument("Unknown activation function type");
    }
}

// Get the derivative of the activation function based on the specified type
std::function<Eigen::VectorXd(const Eigen::VectorXd &)> ActivationFunction::getDerivative(Type type)
{
    if (type < Type::Linear || type > Type::Softmax)
    {
        throw std::invalid_argument("Invalid activation function type");
    }

    switch (type)
    {
    case Type::Linear:
        return [](const Eigen::VectorXd &x) noexcept -> Eigen::VectorXd
        {
            try
            {
                return Eigen::VectorXd::Ones(x.size());
            }
            catch (...)
            {
                std::cerr << "Error in Linear derivative" << std::endl;
                return Eigen::VectorXd::Zero(x.size());
            }
        };
    case Type::ReLU:
        return [](const Eigen::VectorXd &x) noexcept -> Eigen::VectorXd
        {
            try
            {
                return x.cwiseMax(0.0);
            }
            catch (...)
            {
                std::cerr << "Error in ReLU activation" << std::endl;
                return Eigen::VectorXd::Zero(x.size());
            }
        };
    case Type::Sigmoid:
        return [](const Eigen::VectorXd &x) noexcept -> Eigen::VectorXd
        {
            try
            {
                Eigen::VectorXd s = 1.0 / (1.0 + (-x.array().min(50.0).max(-50.0)).exp());
                return s.array() * (1.0 - s.array());
            }
            catch (...)
            {
                std::cerr << "Error in Sigmoid derivative" << std::endl;
                return Eigen::VectorXd::Zero(x.size());
            }
        };
    case Type::Tanh:
        return [](const Eigen::VectorXd &x) noexcept -> Eigen::VectorXd
        {
            try
            {
                Eigen::VectorXd t = x.array().tanh();
                return 1.0 - t.array().square();
            }
            catch (...)
            {
                std::cerr << "Error in Tanh derivative" << std::endl;
                return Eigen::VectorXd::Zero(x.size());
            }
        };
    case Type::Softmax:
        return [](const Eigen::VectorXd &x) noexcept -> Eigen::VectorXd
        {
            try
            {
                Eigen::VectorXd exp_x = (x.array() - x.maxCoeff()).exp();
                Eigen::VectorXd s = exp_x.array() / (exp_x.sum() + std::numeric_limits<double>::epsilon());
                return s.array() * (1.0 - s.array());
            }
            catch (...)
            {
                std::cerr << "Error in Softmax derivative" << std::endl;
                return Eigen::VectorXd::Zero(x.size());
            }
        };
    default:
        throw std::invalid_argument("Unknown activation function type");
    }
}