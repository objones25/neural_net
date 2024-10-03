import sys
print("Python version:", sys.version)
print("sys.path:", sys.path)

try:
    import neural_network_py
    print("Successfully imported neural_network_py")
    print("Contents of neural_network_py:", dir(neural_network_py))
    
    if hasattr(neural_network_py, 'ActivationType'):
        print("ActivationType is present")
        print("ActivationType values:", list(neural_network_py.ActivationType))
    else:
        print("ActivationType is not present in the module")
    
    if hasattr(neural_network_py, 'WeightInitialization'):
        print("WeightInitialization is present")
        print("WeightInitialization values:", list(neural_network_py.WeightInitialization))
    else:
        print("WeightInitialization is not present in the module")
    
    if hasattr(neural_network_py, 'LossFunction'):
        print("LossFunction is present")
        print("LossFunction values:", list(neural_network_py.LossFunction))
    else:
        print("LossFunction is not present in the module")
    
    if hasattr(neural_network_py, 'NeuralNetwork'):
        print("NeuralNetwork class is present")
    else:
        print("NeuralNetwork class is not present in the module")
    
except ImportError as e:
    print("Error importing neural_network_py:", str(e))
except Exception as e:
    print("An error occurred:", str(e))