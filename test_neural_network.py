import neural_network_py as nn
import numpy as np

def test_network(opt_algo):
    try:
        print(f"\nTesting with {opt_algo.name}")
        print(f"OptimizationAlgorithm enum value: {opt_algo.value}")
        network = nn.NeuralNetwork(
            layer_sizes=[2, 3, 1],
            lr=0.01,
            act_func=nn.ActivationFunction.TanH,
            weight_init=nn.NeuralNetwork.WeightInitialization.Xavier,
            opt_algo=opt_algo,
            reg_type=nn.NeuralNetwork.RegularizationType.L2,
            reg_strength=0.01
        )
        print("Neural network created successfully.")

        inputs = [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6]), np.array([0.7, 0.8])]
        targets = [np.array([0.3]), np.array([0.7]), np.array([0.9]), np.array([0.5])]

        print("Training the network...")
        network.train(inputs, targets, epochs=20, batch_size=2, error_tolerance=1e-3)
        print("Training completed.")

        print("Making a prediction...")
        prediction = network.predict(np.array([0.5, 0.6]))
        print(f"Prediction: {prediction}")

        print("Calculating loss...")
        loss = network.get_loss(inputs, targets)
        print(f"Loss: {loss}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

# Print enum values
print(f"GradientDescent enum value: {nn.OptimizationAlgorithm.GradientDescent.value}")
print(f"Adam enum value: {nn.OptimizationAlgorithm.Adam.value}")
print(f"RMSprop enum value: {nn.OptimizationAlgorithm.RMSprop.value}")

# Test all optimization algorithms
test_network(nn.OptimizationAlgorithm.GradientDescent)
test_network(nn.OptimizationAlgorithm.Adam)
test_network(nn.OptimizationAlgorithm.RMSprop)

print("Script execution completed.")