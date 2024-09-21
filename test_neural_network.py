import pytest
import numpy as np
import neural_network_py as nn

def test_network_initialization():
    # Test various network configurations
    network = nn.NeuralNetwork([2, 3, 1])
    assert len(network.layers) == 3
    assert network.layers == [2, 3, 1]

    # Test invalid configurations
    with pytest.raises(nn.NetworkConfigurationError):
        nn.NeuralNetwork([])  # Empty layer list
    with pytest.raises(nn.NetworkConfigurationError):
        nn.NeuralNetwork([1])  # Single layer
    with pytest.raises(nn.NetworkConfigurationError):
        nn.NeuralNetwork([2, 0, 1])  # Layer with zero neurons

def test_activation_functions():
    input_vector = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    # Test ReLU
    network = nn.NeuralNetwork([5, 1], hidden_activation=nn.ActivationType.ReLU, output_activation=nn.ActivationType.Linear)
    output = network.predict(input_vector)
    assert np.allclose(output, np.maximum(input_vector, 0))
    
    # Test Sigmoid
    network = nn.NeuralNetwork([5, 1], hidden_activation=nn.ActivationType.Sigmoid)
    output = network.predict(input_vector)
    expected = 1 / (1 + np.exp(-input_vector))
    assert np.allclose(output, expected)
    
    # Test Tanh
    network = nn.NeuralNetwork([5, 1], hidden_activation=nn.ActivationType.Tanh)
    output = network.predict(input_vector)
    assert np.allclose(output, np.tanh(input_vector))
    
    # Test Linear
    network = nn.NeuralNetwork([5, 1], hidden_activation=nn.ActivationType.Linear)
    output = network.predict(input_vector)
    assert np.allclose(output, input_vector)

    # Test Softmax
    network = nn.NeuralNetwork([5, 3], output_activation=nn.ActivationType.Softmax)
    output = network.predict(input_vector)
    assert np.allclose(np.sum(output), 1.0)
    assert np.all(output >= 0) and np.all(output <= 1)

def test_weight_initialization():
    np.random.seed(42)  # For reproducibility
    
    # Test Random initialization
    network = nn.NeuralNetwork([100, 50, 10], weight_init=nn.NeuralNetwork.WeightInitialization.Random)
    assert -1 < np.mean(network.weights[0]) < 1 and 0 < np.std(network.weights[0]) < 1
    
    # Test Xavier initialization
    network = nn.NeuralNetwork([100, 50, 10], weight_init=nn.NeuralNetwork.WeightInitialization.Xavier)
    expected_std = np.sqrt(2 / (100 + 50))
    assert np.isclose(np.std(network.weights[0]), expected_std, atol=0.1)
    
    # Test He initialization
    network = nn.NeuralNetwork([100, 50, 10], weight_init=nn.NeuralNetwork.WeightInitialization.He)
    expected_std = np.sqrt(2 / 100)
    assert np.isclose(np.std(network.weights[0]), expected_std, atol=0.1)

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
        l1_sparsity = np.mean(np.abs(network_l1.weights[0]) < 1e-3)
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
        l2_sparsity = np.mean(np.abs(network_l2.weights[0]) < 1e-3)
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
    network = nn.NeuralNetwork([2, 10, 5, 1], learning_rate=0.01)
    X = np.array([[1, 2]])
    y = np.array([[3]])
    
    initial_loss = network.get_loss(X, y)
    network.train(X, y, epochs=1000, batch_size=1)
    final_loss = network.get_loss(X, y)
    
    assert final_loss < initial_loss

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

def test_network_creation_with_regularization():
    print("Testing network creation with regularization")
    try:
        nn.NeuralNetwork([2, 5, 1], reg_type=nn.NeuralNetwork.RegularizationType.L1, reg_strength=0.1)
        nn.NeuralNetwork([2, 5, 1], reg_type=nn.NeuralNetwork.RegularizationType.L2, reg_strength=0.1)
        print("Network creation successful")
    except Exception as e:
        pytest.fail(f"Network creation failed: {str(e)}")

def test_single_update_with_regularization():
    print("Testing single update with regularization")
    np.random.seed(42)
    X = np.random.randn(1, 2)
    y = np.random.randn(1, 1)
    
    for reg_type in [nn.NeuralNetwork.RegularizationType.L1, nn.NeuralNetwork.RegularizationType.L2]:
        print(f"Testing {reg_type}")
        network = nn.NeuralNetwork([2, 5, 1], reg_type=reg_type, reg_strength=0.1)
        try:
            initial_loss = network.get_loss(X, y)
            network.train(X, y, epochs=1, batch_size=1)
            final_loss = network.get_loss(X, y)
            print(f"Initial loss: {initial_loss}, Final loss: {final_loss}")
            assert final_loss <= initial_loss, f"Loss did not decrease for {reg_type}"
        except Exception as e:
            pytest.fail(f"Single update failed for {reg_type}: {str(e)}")

def test_regularization_effect():
    print("Testing regularization effect")
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.random.randn(100, 1)
    
    networks = {
        "No Regularization": nn.NeuralNetwork([2, 5, 1], reg_type=nn.NeuralNetwork.RegularizationType.NONE, reg_strength=0.0),
        "L1": nn.NeuralNetwork([2, 5, 1], reg_type=nn.NeuralNetwork.RegularizationType.L1, reg_strength=0.1),
        "L2": nn.NeuralNetwork([2, 5, 1], reg_type=nn.NeuralNetwork.RegularizationType.L2, reg_strength=0.1)
    }
    
    for name, network in networks.items():
        try:
            print(f"Training {name}")
            network.train(X, y, epochs=100, batch_size=32)
            weights = network.weights[0]
            sparsity = np.mean(np.abs(weights) < 1e-3)
            print(f"{name} - Sparsity: {sparsity}, Mean weight: {np.mean(np.abs(weights))}")
        except Exception as e:
            pytest.fail(f"Training failed for {name}: {str(e)}")
    
    l1_sparsity = np.mean(np.abs(networks["L1"].weights[0]) < 1e-3)
    l2_sparsity = np.mean(np.abs(networks["L2"].weights[0]) < 1e-3)
    assert l1_sparsity > l2_sparsity, "L1 regularization should induce more sparsity than L2"

if __name__ == "__main__":
    pytest.main([__file__])

