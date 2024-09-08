from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'neural_network_cpp',
        ['neural_network.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++11']
    ),
]

setup(
    name='neural_network_cpp',
    version='0.0.1',
    author='Your Name',
    description='A neural network implementation with C++ backend',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    install_requires=['pybind11>=2.5.0'],
)