from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

# Correct Eigen path for Mac installed via Homebrew
EIGEN_PATH = "/usr/local/Cellar/eigen/3.4.0_1/include/eigen3"

ext_modules = [
    Pybind11Extension("neural_network_py",
        ["neural_network_binding.cpp", "neural_network.cpp", "optimization_algorithms.cpp", "activation_functions.cpp"],
        include_dirs=[
            pybind11.get_include(), 
            EIGEN_PATH
        ],
        extra_compile_args=["-std=c++17", f"-I{EIGEN_PATH}"],
        cxx_std=17,
    ),
]

setup(
    name="neural_network_py",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)