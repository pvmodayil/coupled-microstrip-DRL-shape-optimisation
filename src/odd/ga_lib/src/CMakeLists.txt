cmake_minimum_required(VERSION 3.10)

message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")

# Set the project name
project(CoupledStripShapeOptimization)

# Set the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable compiler optimizations (e.g., -O3 for maximum optimization)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# For OpenMP (optional)
find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")

# Add pybind11 from local extern folder
# Find Python interpreter and development libraries from your venv automatically
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

message(STATUS "Using Python executable: ${Python3_EXECUTABLE}")
add_subdirectory(extern/pybind11)

# Build the Python module
pybind11_add_module(ga_cpp
    ga_wrapper.cpp
    include/coupledstrip_lib.cpp
    include/genetic_algorithm.cpp
)

target_include_directories(ga_cpp PRIVATE include extern)

# Link OpenMP if you use it
if(OpenMP_CXX_FOUND)
    target_link_libraries(ga_cpp PUBLIC OpenMP::OpenMP_CXX)
endif()