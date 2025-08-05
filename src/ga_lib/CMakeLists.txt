cmake_minimum_required(VERSION 3.10)

# Set the project name
project(CoupledStripShapeOptimization)

# Set the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable compiler optimizations (e.g., -O3 for maximum optimization)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

add_executable(coupledstrip_arrangement
  main.cpp
  include/file_io.cpp
  include/coupledstrib_lib.cpp
  include/genetic_algorithm.cpp
)

target_include_directories(coupledstrip_arrangement PRIVATE include extern)

# Enable OpenMP if available
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(coupledstrip_arrangement PUBLIC OpenMP::OpenMP_CXX)
endif()