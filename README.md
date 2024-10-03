# Neural Network Python Library

This library provides a simple neural network implementation with Python bindings to C++.

## Installation

```bash
pip install .
```

## Usage

```python
import neural_network_py as nn
import numpy as np

# Create a neural network
network = nn.NeuralNetwork(
    layer_sizes=[2, 5, 1],
    hidden_activation=nn.ActivationType.ReLU,
    output_activation=nn.ActivationType.Sigmoid,
    optimizer_name="Adam",
    learning_rate=0.01,
    loss_function=nn.LossFunction.MeanSquaredError
)

# Generate some dummy data
X = np.random.rand(100, 2)
y = np.random.rand(100, 1)

# Train the network
network.train(X, y, epochs=100, batch_size=32, learning_rate=0.01)

# Make predictions
X_test = np.random.rand(10, 2)
predictions = network.predict(X_test)

print(predictions)
```

## API Reference

### NeuralNetwork

- `__init__(layer_sizes, hidden_activation, output_activation, optimizer_name, learning_rate, loss_function, use_batch_norm, weight_init)`
- `predict(input_data)`
- `train(X, y, epochs, batch_size, learning_rate)`
- `calculate_loss(X, y)`
- `set_learning_rate(learning_rate)`
- `get_learning_rate()`

### Enums

- `ActivationType`: Linear, ReLU, Sigmoid, Tanh, Softmax
- `WeightInitialization`: Xavier, He, LeCun
- `LossFunction`: MeanSquaredError, CrossEntropy