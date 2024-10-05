import sys
import pytest
import numpy as np

def import_nn():
    try:
        import neural_network_py as nn
        return nn
    except ImportError as e:
        print("Error importing neural_network_py:", str(e))
        return None

@pytest.fixture(scope="module")
def nn():
    return import_nn()

def test_enums(nn):
    assert nn is not None
    assert hasattr(nn, 'ActivationType')
    assert hasattr(nn, 'WeightInitialization')
    assert hasattr(nn, 'LossFunction')
    assert hasattr(nn.ActivationType, 'ReLU')
    assert hasattr(nn.WeightInitialization, 'Xavier')
    assert hasattr(nn.LossFunction, 'MeanSquaredError')

@pytest.fixture
def simple_network(nn):
    if nn is None:
        pytest.skip("neural_network_py module not available")
    return nn.NeuralNetwork(
        [2, 3, 1],
        nn.ActivationType.ReLU,
        nn.ActivationType.Sigmoid,
        "Adam",
        0.01,
        nn.LossFunction.MeanSquaredError,
        True,
        nn.WeightInitialization.Xavier
    )

def test_network_initialization(simple_network):
    assert simple_network is not None
    assert simple_network.get_learning_rate() == 0.01

def test_prediction_shape(simple_network):
    input_data = np.random.rand(1, 2)
    prediction = simple_network.predict(input_data)
    assert prediction.shape == (1, 1)

def test_training(nn):
    if nn is None:
        pytest.skip("neural_network_py module not available")
    network = nn.NeuralNetwork(
        [2, 5, 1],
        nn.ActivationType.ReLU,
        nn.ActivationType.Sigmoid,
        "Adam",
        0.01,
        nn.LossFunction.MeanSquaredError,
        True,
        nn.WeightInitialization.Xavier
    )
    nn.NeuralNetwork.enable_debug_logging(True)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).reshape(-1, 1).astype(float)
    
    initial_loss = network.calculate_loss(X, y)
    # Updated train call with additional parameters
    network.train(X, y, 100, 32, 0.01, 0.2, 5, 1e-4)
    final_loss = network.calculate_loss(X, y)
    
    assert final_loss < initial_loss

def test_learning_rate_update(simple_network):
    initial_lr = simple_network.get_learning_rate()
    new_lr = 0.001
    simple_network.set_learning_rate(new_lr)
    assert simple_network.get_learning_rate() == new_lr
    assert simple_network.get_learning_rate() != initial_lr

def test_xor_problem(nn):
    if nn is None:
        pytest.skip("neural_network_py module not available")
    network = nn.NeuralNetwork(
        [2, 5, 1],
        nn.ActivationType.ReLU,
        nn.ActivationType.Sigmoid,
        "Adam",
        0.01,
        nn.LossFunction.MeanSquaredError,
        True,
        nn.WeightInitialization.Xavier
    )
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    network.train(X, y, epochs=1000, batch_size=4, learning_rate=0.01)
    
    predictions = network.predict(X)
    rounded_predictions = np.round(predictions)
    
    assert np.allclose(rounded_predictions, y, atol=0.1)

def test_invalid_network_configuration(nn):
    with pytest.raises(Exception):  # Replace with your specific error type if available
        nn.NeuralNetwork(
            [],  # Empty layer sizes
            nn.ActivationType.ReLU,
            nn.ActivationType.Sigmoid,
            "Adam",
            0.01,
            nn.LossFunction.MeanSquaredError,
            True,
            nn.WeightInitialization.Xavier
        )

def test_different_optimizers(nn):
    optimizers = ["Adam", "RMSprop"]
    for opt in optimizers:
        network = nn.NeuralNetwork(
            [2, 3, 1],
            nn.ActivationType.ReLU,
            nn.ActivationType.Sigmoid,
            opt,
            0.01,
            nn.LossFunction.MeanSquaredError,
            True,
            nn.WeightInitialization.Xavier
        )
        assert network is not None

def test_numerical_stability(nn):
    if nn is None:
        pytest.skip("neural_network_py module not available")
    network = nn.NeuralNetwork(
        [2, 5, 1],
        nn.ActivationType.ReLU,
        nn.ActivationType.Sigmoid,
        "Adam",
        0.01,
        nn.LossFunction.MeanSquaredError,
        True,
        nn.WeightInitialization.Xavier
    )
    
    X = np.random.rand(100, 2)
    y = np.random.rand(100, 1)
    
    initial_loss = network.calculate_loss(X, y)
    network.train(X, y, epochs=10, batch_size=32, learning_rate=0.01)
    final_loss = network.calculate_loss(X, y)
    
    assert np.isfinite(initial_loss)
    assert np.isfinite(final_loss)

def test_debug_logging(nn, capsys):
    if nn is None:
        pytest.skip("neural_network_py module not available")
    
    nn.NeuralNetwork.enable_debug_logging(True)
    
    network = nn.NeuralNetwork(
        [2, 3, 1],
        nn.ActivationType.ReLU,
        nn.ActivationType.Sigmoid,
        "Adam",
        0.01,
        nn.LossFunction.MeanSquaredError,
        True,
        nn.WeightInitialization.Xavier
    )
    
    X = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    
    network.train(X, y, epochs=1, batch_size=5, learning_rate=0.01)
    
    captured = capsys.readouterr()
    assert "DEBUG" in captured.out
    assert "INFO" in captured.out

def test_early_stopping(nn):
    if nn is None:
        pytest.skip("neural_network_py module not available")
    
    network = nn.NeuralNetwork(
        [2, 5, 1],
        nn.ActivationType.ReLU,
        nn.ActivationType.Sigmoid,
        "Adam",
        0.01,
        nn.LossFunction.MeanSquaredError,
        True,
        nn.WeightInitialization.Xavier
    )
    
    X = np.random.rand(100, 2)
    y = np.random.rand(100, 1)
    
    network.train(X, y, epochs=1000, batch_size=32, learning_rate=0.01, validation_split=0.2, patience=5, min_delta=1e-4)
    
    assert True  # If we reach this point without exceptions, the test passes