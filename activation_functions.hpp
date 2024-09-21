#pragma once

#include <Eigen/Dense>
#include <functional>

class ActivationFunction {
public:
    enum class Type {
        Linear,
        ReLU,
        Sigmoid,
        Tanh,
        Softmax
    };

    ActivationFunction(Type hiddenType, Type outputType);

    Eigen::VectorXd activateHidden(const Eigen::VectorXd& x) const;
    Eigen::VectorXd derivativeHidden(const Eigen::VectorXd& x) const;
    Eigen::VectorXd activateOutput(const Eigen::VectorXd& x) const;
    Eigen::VectorXd derivativeOutput(const Eigen::VectorXd& x) const;

    Type getOutputActivationType() const { return outputType; }

private:
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> hiddenActivation;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> hiddenDerivative;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> outputActivation;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> outputDerivative;

    static std::function<Eigen::VectorXd(const Eigen::VectorXd&)> getFunction(Type type);
    static std::function<Eigen::VectorXd(const Eigen::VectorXd&)> getDerivative(Type type);
    Type outputType;
};