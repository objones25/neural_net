import sys
print("Python version:", sys.version)
print("sys.path:", sys.path)

try:
    import neural_network_py
    print("Imported neural_network_py successfully")
    print("Dir of neural_network_py:", dir(neural_network_py))
    print("ActivationType:", neural_network_py.ActivationType)
    print("WeightInitialization:", neural_network_py.WeightInitialization)
    print("LossFunction:", neural_network_py.LossFunction)
    print("NeuralNetwork:", neural_network_py.NeuralNetwork)
    print("test_enums:", neural_network_py.test_enums)
    neural_network_py.test_enums()
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()