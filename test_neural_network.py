import pytest
import numpy as np
import neural_network_py as nn

def test_network_initialization():
    # Test various network configurations
    network = nn.NeuralNetwork([2, 3, 1])
    assert len(network.getLayers()) == 2  # Number of weight matrices
    assert [layer.weights.shape for layer in network.getLayers()] == [(3, 2), (1, 3)]

    # Test invalid configurations
    with pytest.raises(nn.NetworkConfigurationError):
        nn.NeuralNetwork([])  # Empty layer list
    with pytest.raises(nn.NetworkConfigurationError):
        nn.NeuralNetwork([1])  # Single layer
    with pytest.raises(nn.NetworkConfigurationError):
        nn.NeuralNetwork([2, 0, 1])  # Layer with zero neurons

def test_activation_functions():
    print("\nTesting activation functions")
    input_vector = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    activation_types = [
        (nn.ActivationType.ReLU, lambda x: np.maximum(x, 0)),
        (nn.ActivationType.Sigmoid, lambda x: 1 / (1 + np.exp(-x))),
        (nn.ActivationType.Tanh, np.tanh),
        (nn.ActivationType.Linear, lambda x: x),
    ]

    for act_type, expected_func in activation_types:
        print(f"\nTesting {act_type}")

        network = nn.NeuralNetwork([5, 5, 5],
                                   hidden_activation=act_type,
                                   output_activation=nn.ActivationType.Linear,
                                   use_batch_norm=False)
        print(f"Network created with hidden activation: {act_type}, output activation: Linear")

        # Set weights and biases
        weights = [np.eye(5), np.eye(5)]
        biases = [np.zeros(5), np.zeros(5)]
        network.set_weights(weights)
        network.set_biases(biases)

        output = network.predict(input_vector)
        expected_output = expected_func(input_vector)

        print(f"Input: {input_vector}")
        print(f"Output: {output}")
        print(f"Expected: {expected_output}")

        assert np.allclose(output, expected_output, atol=1e-5), \
            f"{act_type} activation test failed. Output: {output}, Expected: {expected_output}"

    # Test Softmax
    print("\nTesting Softmax")
    network_softmax = nn.NeuralNetwork([5, 5], 
                                       hidden_activation=nn.ActivationType.Linear, 
                                       output_activation=nn.ActivationType.Softmax,
                                       use_batch_norm=False)
    network_softmax.set_weights([np.eye(5)])
    network_softmax.set_biases([np.zeros(5)])
    
    output_softmax = network_softmax.predict(input_vector)
    exp_values = np.exp(input_vector - np.max(input_vector))
    expected_output_softmax = exp_values / np.sum(exp_values)
    
    print(f"Softmax - Input: {input_vector}")
    print(f"Softmax - Output: {output_softmax}")
    print(f"Softmax - Expected: {expected_output_softmax}")
    
    assert np.allclose(output_softmax, expected_output_softmax, atol=1e-5), \
        f"Softmax test failed. Output: {output_softmax}, Expected: {expected_output_softmax}"
    assert np.isclose(np.sum(output_softmax), 1.0), \
        f"Softmax output does not sum to 1. Sum: {np.sum(output_softmax)}"

def test_weight_initialization():
    np.random.seed(42)
    
    # Test Random initialization
    network = nn.NeuralNetwork([100, 50, 10], weight_init=nn.NeuralNetwork.WeightInitialization.Random)
    weights = network.getLayers()[0].weights
    assert -1 < np.mean(weights) < 1 and 0 < np.std(weights) < 1
    
    # Test Xavier initialization
    network = nn.NeuralNetwork([100, 50, 10], weight_init=nn.NeuralNetwork.WeightInitialization.Xavier)
    weights = network.getLayers()[0].weights
    expected_std = np.sqrt(2 / (100 + 50))
    assert np.isclose(np.std(weights), expected_std, atol=0.1)
    
    # Test He initialization
    network = nn.NeuralNetwork([100, 50, 10], weight_init=nn.NeuralNetwork.WeightInitialization.He)
    weights = network.getLayers()[0].weights
    expected_std = np.sqrt(2 / 100)
    assert np.isclose(np.std(weights), expected_std, atol=0.1)

def test_regularization():
    print("Starting test_regularization")
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.random.randn(100, 1)
    
    print("Creating L1 network")
    network_l1 = nn.NeuralNetwork([2, 5, 1], reg_type=nn.NeuralNetwork.RegularizationType.L1, reg_strength=0.1)
    print("L1 network created")
    
    try:
        print("Training L1 network")
        network_l1.train(X, y, epochs=100, batch_size=32)
        print("L1 network trained")
        
        print("Calculating L1 sparsity")
        l1_sparsity = np.mean(np.abs(network_l1.getLayers()[0].weights) < 1e-3)
        print(f"L1 sparsity: {l1_sparsity}")
    except Exception as e:
        print(f"Error during L1 network operations: {str(e)}")
        raise
    
    print("Creating L2 network")
    network_l2 = nn.NeuralNetwork([2, 5, 1], reg_type=nn.NeuralNetwork.RegularizationType.L2, reg_strength=0.1)
    print("L2 network created")
    
    try:
        print("Training L2 network")
        network_l2.train(X, y, epochs=100, batch_size=32)
        print("L2 network trained")
        
        print("Calculating L2 sparsity")
        l2_sparsity = np.mean(np.abs(network_l2.getLayers()[0].weights) < 1e-3)
        print(f"L2 sparsity: {l2_sparsity}")
    except Exception as e:
        print(f"Error during L2 network operations: {str(e)}")
        raise
    
    # L1 should induce more sparsity than L2
    assert l1_sparsity > l2_sparsity
    print("Test completed successfully")

def test_optimizers():
    np.random.seed(42)
    X = np.random.randn(1000, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    
    optimizers = ["GradientDescent", "Adam", "RMSprop"]
    losses = {}
    
    for opt in optimizers:
        network = nn.NeuralNetwork([2, 10, 1], optimizer_name=opt, learning_rate=0.01)
        network.train(X, y, epochs=1000, batch_size=32)
        losses[opt] = network.get_loss(X, y)
    
    # Check if all optimizers converged
    assert all(loss < 0.1 for loss in losses.values())
    
    # Check if adaptive optimizers (Adam, RMSprop) performed better than GradientDescent
    assert losses["GradientDescent"] > losses["Adam"]
    assert losses["GradientDescent"] > losses["RMSprop"]

def test_backpropagation():
    np.random.seed(42)
    network = nn.NeuralNetwork([2, 10, 5, 1], learning_rate=0.01, use_batch_norm=True)
    X = np.array([[1, 2]])
    y = np.array([[3]])
    
    initial_loss = network.get_loss(X, y)
    print(f"Initial loss: {initial_loss}")

    # Perform one training step
    network.train(X, y, epochs=1, batch_size=1)
    
    # Check if gradients for batch norm parameters exist
    for i, layer in enumerate(network.getLayers()[:-1], 1):  # Exclude output layer
        if layer.batch_norm is not None:
            assert hasattr(layer, 'bn_gamma_grad'), f"Batch norm gamma gradient not computed for layer {i}"
            assert hasattr(layer, 'bn_beta_grad'), f"Batch norm beta gradient not computed for layer {i}"
            print(f"Batch norm gradients for layer {i}:")
            print(f"Gamma grad: {layer.bn_gamma_grad}")
            print(f"Beta grad: {layer.bn_beta_grad}")
        else:
            print(f"Layer {i} does not have batch normalization")

    final_loss = network.get_loss(X, y)
    print(f"Final loss: {final_loss}")
    
    assert final_loss < initial_loss, "Loss did not decrease after training step"

def test_numerical_stability():
    # Test with very large inputs
    network = nn.NeuralNetwork([2, 3, 1])
    large_input = np.array([1e10, 1e10])
    output = network.predict(large_input)
    assert np.isfinite(output).all()
    
    # Test with very small inputs
    small_input = np.array([1e-10, 1e-10])
    output = network.predict(small_input)
    assert np.isfinite(output).all()

def test_batch_processing():
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
    
    network_batch = nn.NeuralNetwork([2, 5, 1])
    network_batch.train(X, y, epochs=10, batch_size=32)
    
    network_full = nn.NeuralNetwork([2, 5, 1])
    network_full.train(X, y, epochs=10, batch_size=100)
    
    assert abs(network_batch.get_loss(X, y) - network_full.get_loss(X, y)) < 0.1

if __name__ == "__main__":
    pytest.main([__file__])