from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

# Adjust this path to match your Eigen installation
EIGEN_PATH = "/usr/local/opt/eigen/include/eigen3"

ext_modules: list[Pybind11Extension] = [
    Pybind11Extension("neural_network_py",
        ["neural_network_binding.cpp", 
         "neural_network_backward.cpp",
         "neural_network_core.cpp",
         "neural_network_forward.cpp",
         "neural_network_prediction.cpp",
         "neural_network_training.cpp",
         "neural_network_utils.cpp", 
         "optimization_algorithms.cpp", 
         "activation_functions.cpp",
         "layer.cpp"],  # Add this line
        include_dirs=[
            pybind11.get_include(), 
            EIGEN_PATH
        ],
        extra_compile_args=["-std=c++17", f"-I{EIGEN_PATH}", "-g", "-O0", "-fno-inline"],
        extra_link_args=["-g"],
        cxx_std=17,
    ),
]

setup(
    name="neural_network_py",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)