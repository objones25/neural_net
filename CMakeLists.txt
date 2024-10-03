cmake_minimum_required(VERSION 3.15)
project(neural_network_py)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Scikit-Build specific configuration
if(SKBUILD)
  set(SKBUILD_SCRIPTS_DIR ${SKBUILD_DIR}/cmake)
  list(APPEND CMAKE_MODULE_PATH ${SKBUILD_SCRIPTS_DIR})
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Find Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add your source files
set(SOURCES
    src/neural_network_binding.cpp
    src/neural_network.cpp
    src/layer.cpp
    src/activation_functions.cpp
    src/performance_tuning.cpp
    src/logger.cpp
)

# Create the pybind11 module
pybind11_add_module(neural_network_py ${SOURCES})

# Include directories
target_include_directories(neural_network_py PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(neural_network_py PRIVATE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(neural_network_py PRIVATE OpenMP::OpenMP_CXX)
endif()

# Set installation directory
if(NOT DEFINED SKBUILD_CORE_INSTALL_DIR)
  set(SKBUILD_CORE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
endif()

# Installation
install(TARGETS neural_network_py 
        DESTINATION ${SKBUILD_CORE_INSTALL_DIR}
        COMPONENT python)