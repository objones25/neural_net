import sys
import numpy as np

print("Python version:", sys.version)
print("sys.path:", sys.path)

try:
    import neural_network_py as nn
    import numpy as np
    
    print("neural_network_py imported successfully")
    print("ActivationType:", nn.ActivationType)
    
    # Check if enable_debug_logging exists
    if hasattr(nn.NeuralNetwork, 'enable_debug_logging'):
        nn.NeuralNetwork.enable_debug_logging(True)
        print("Debug logging enabled")
    else:
        print("Warning: enable_debug_logging method not found")
    
    # Test creating a neural network
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
    print("NeuralNetwork object created successfully")
    
    # Test getting learning rate
    print("Learning rate:", network.get_learning_rate())
    
    # Test prediction
    test_input = np.random.rand(1, 2)
    print("Input shape:", test_input.shape)
    print("Input size:", network.get_input_size())
    prediction = network.predict(test_input)
    print("Prediction shape:", prediction.shape)
    
    print("All basic functionality tests passed")
except ImportError as e:
    print("Error importing neural_network_py:", str(e))
    import traceback
    traceback.print_exc()
except Exception as e:
    print("Error during functionality test:", str(e))
    import traceback
    traceback.print_exc()