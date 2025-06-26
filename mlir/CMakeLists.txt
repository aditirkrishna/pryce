# CMake configuration for DerivLab MLIR dialect

cmake_minimum_required(VERSION 3.13)
project(DerivLabDialect)

find_package(MLIR REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_library(DerivLabDialect
  lib/Dialect/DerivLabDialect.cpp
)

target_link_libraries(DerivLabDialect
  PRIVATE MLIRIR
)

add_library(DerivLabTransforms
  lib/Transforms/LowerSimulateGBM.cpp
  lib/Transforms/LowerPayoff.cpp
)

target_link_libraries(DerivLabTransforms
  PRIVATE DerivLabDialect MLIRIR
)

# Optionally add tests and tools here
