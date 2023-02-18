cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cpp_api_example)

# API Download: https://pytorch.org/cppdocs/installing.html

set(CMAKE_CUDA_COMPILER /usr/local/cuda12.0/bin/nvcc)

# Find the Torch package
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Create an executable
add_executable(cpp_api_example src/cpp_api_example.cpp)

# Link against the PyTorch libraries
target_link_libraries(cpp_api_example "${TORCH_LIBRARIES}" "${TORCH_CUDA_LIBRARIES}")

# Set the C++ standard to C++17
set_property(TARGET cpp_api_example PROPERTY CXX_STANDARD 17)