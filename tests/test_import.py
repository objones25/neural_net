import neural_network_py as nn

print("Imported neural_network_py successfully")
print("ActivationType:", nn.ActivationType)
print("WeightInitialization:", nn.WeightInitialization)
print("LossFunction:", nn.LossFunction)

# Try creating a neural network
network = nn.NeuralNetwork(
    [2, 3, 1],  # layer_sizes
    nn.ActivationType.ReLU,  # hidden_activation
    nn.ActivationType.Sigmoid,  # output_activation
    "Adam",  # optimizer_name
    0.01,  # learning_rate
    nn.LossFunction.MeanSquaredError,
    True,  # use_batch_norm (default value)
    nn.WeightInitialization.Xavier  # weight_init (default value)
)
print("Created NeuralNetwork successfully")