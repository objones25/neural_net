import sys
import pytest
import numpy as np

print("Python version:", sys.version)
print("sys.path:", sys.path)

def import_nn():
    try:
        import neural_network_py as nn
        print("neural_network_py imported successfully")
        print("Dir of nn:", dir(nn))
        print("ActivationType:", getattr(nn, 'ActivationType', 'Not found'))
        print("WeightInitialization:", getattr(nn, 'WeightInitialization', 'Not found'))
        print("LossFunction:", getattr(nn, 'LossFunction', 'Not found'))
        
        # Test the enums
        nn.test_enums()
        
        return nn
    except ImportError as e:
        print("Error importing neural_network_py:", str(e))
        import traceback
        traceback.print_exc()
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
        [2, 3, 1],  # layer_sizes
        nn.ActivationType.ReLU,
        nn.ActivationType.Sigmoid,
        "Adam",
        0.01,
        nn.LossFunction.MeanSquaredError,
        True,  # use_batch_norm
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
    
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).reshape(-1, 1).astype(float)
    
    initial_loss = network.calculate_loss(X, y)
    network.train(X, y, epochs=100, batch_size=32, learning_rate=0.01)
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
    with pytest.raises(Exception):  # Replace with your specific error type
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
    optimizers = ["Adam", "RMSprop"]  # Add more if you have implemented others
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
