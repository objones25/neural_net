#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "activation_functions.hpp"
#include "optimization_algorithms.hpp"
#include "batch_normalization.hpp"
#include "exceptions.hpp"
#include "clip_and_check.hpp"
#include <algorithm>
#include <random>
#include <numeric>
#include <iostream>

#define DEBUG_LOG(x) std::cout << "[DEBUG] " << x << std::endl

class NeuralNetwork; // Forward declaration