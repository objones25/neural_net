import cProfile
import pstats
import time
import numpy as np
import neural_network_py as nn
from memory_profiler import profile

def generate_dataset(size=10000, input_dim=2):
    np.random.seed(42)
    inputs = np.random.randn(size, input_dim)
    targets = np.sum(inputs, axis=1, keepdims=True)
    return inputs, targets

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@time_function
def profile_network_creation(layer_sizes):
    return nn.NeuralNetwork(layer_sizes)

@time_function
def profile_prediction(network, inputs):
    for input_data in inputs:
        network.predict(input_data)

@profile
def profile_training(network, inputs, targets, epochs, batch_size):
    network.train(inputs, targets, epochs=epochs, batch_size=batch_size)

def profile_different_sizes():
    print("\nProfiling networks of different sizes:")
    sizes = [(2, 5, 1), (2, 10, 10, 1), (10, 50, 50, 10)]
    for size in sizes:
        print(f"\nNetwork size: {size}")
        network = profile_network_creation(size)
        inputs, targets = generate_dataset(size=1000, input_dim=size[0])
        profile_prediction(network, inputs)
        profile_training(network, inputs, targets, epochs=10, batch_size=32)

def profile_optimizers():
    print("\nProfiling different optimizers:")
    optimizers = ["GradientDescent", "Adam", "RMSprop"]
    inputs, targets = generate_dataset(size=5000)
    for opt in optimizers:
        print(f"\nOptimizer: {opt}")
        network = nn.NeuralNetwork([2, 10, 1], optimizer_name=opt)
        profile_training(network, inputs, targets, epochs=50, batch_size=32)

def profile_batch_sizes():
    print("\nProfiling different batch sizes:")
    batch_sizes = [1, 32, 128, 512]
    inputs, targets = generate_dataset(size=10000)
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        network = nn.NeuralNetwork([2, 10, 1])
        profile_training(network, inputs, targets, epochs=10, batch_size=batch_size)

def main():
    # Profile network creation and prediction
    inputs, targets = generate_dataset()
    network = profile_network_creation([2, 10, 1])
    profile_prediction(network, inputs)

    # Profile training
    print("\nProfiling training:")
    profiler = cProfile.Profile()
    profiler.enable()
    profile_training(network, inputs, targets, epochs=10, batch_size=32)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)  # Print top 20 time-consuming operations

    # Profile networks of different sizes
    profile_different_sizes()

    # Profile different optimizers
    profile_optimizers()

    # Profile different batch sizes
    profile_batch_sizes()

if __name__ == "__main__":
    main()