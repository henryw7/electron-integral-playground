cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(electron-integral-playground C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

add_compile_options(-Wall -Wno-unused-function -O3 -fPIC)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")

add_subdirectory(electron_integral_playground/c_library)
add_subdirectory(electron_integral_playground/c_library/molecular_integral)
