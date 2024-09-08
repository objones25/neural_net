import numpy as np
import neural_network_cpp

class NeuralNetwork:
    def __init__(self, layer_sizes, activation="sigmoid", use_softmax=False):
        """
        Initialize the Neural Network.
        
        :param layer_sizes: List of integers representing the number of neurons in each layer.
        :param activation: Activation function to use ("sigmoid", "relu", or "tanh").
        :param use_softmax: Whether to use softmax activation in the output layer.
        """
        self.nn = neural_network_cpp.NeuralNetwork(layer_sizes, activation, use_softmax)

    def forward(self, input_data):
        """
        Perform a forward pass through the network.
        
        :param input_data: Input data as a numpy array.
        :return: Output of the network as a numpy array.
        """
        return np.array(self.nn.forward(input_data.tolist()))

    def train(self, inputs, targets, epochs, learning_rate, batch_size, num_threads):
        """
        Train the neural network using parallel batch stochastic gradient descent.
        
        :param inputs: Training inputs as a numpy array.
        :param targets: Training targets as a numpy array.
        :param epochs: Number of training epochs.
        :param learning_rate: Learning rate for gradient descent.
        :param batch_size: Size of batches for batch SGD.
        :param num_threads: Number of threads to use for parallel processing.
        """
        self.nn.train(inputs.tolist(), targets.tolist(), epochs, learning_rate, batch_size, num_threads)

    def set_activation(self, activation):
        """
        Set the activation function for the network.
        
        :param activation: Activation function name ("sigmoid", "relu", or "tanh").
        """
        self.nn.set_activation(activation)

    def get_activation(self):
        """
        Get the current activation function of the network.
        
        :return: Name of the current activation function.
        """
        return self.nn.get_activation()

    def set_softmax_output(self, use_softmax):
        """
        Set whether to use softmax activation in the output layer.
        
        :param use_softmax: Boolean indicating whether to use softmax.
        """
        self.nn.set_softmax_output(use_softmax)

    def get_softmax_output(self):
        """
        Check if softmax is being used in the output layer.
        
        :return: Boolean indicating whether softmax is being used.
        """
        return self.nn.get_softmax_output()

# Example usage
if __name__ == "__main__":
    # Create a neural network with 2 input neurons, 3 hidden neurons, and 1 output neuron
    nn = NeuralNetwork([2, 3, 1], activation="relu", use_softmax=False)

    # Generate some dummy data
    np.random.seed(0)
    inputs = np.random.rand(1000, 2)
    targets = np.sum(inputs, axis=1, keepdims=True) > 1

    # Train the network
    nn.train(inputs, targets, epochs=100, learning_rate=0.01, batch_size=32, num_threads=4)

    # Test the network
    test_input = np.array([[0.5, 0.5]])
    output = nn.forward(test_input)
    print(f"Input: {test_input}, Output: {output}")

    # Demonstrate getter and setter methods
    print(f"Current activation: {nn.get_activation()}")
    nn.set_activation("sigmoid")
    print(f"New activation: {nn.get_activation()}")
    print(f"Using softmax output: {nn.get_softmax_output()}")