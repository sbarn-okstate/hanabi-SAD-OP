# Project configuration
cmake_minimum_required(VERSION 3.12)
project(hanabi_learning_environment)

# Set compiler flags
set(CMAKE_C_FLAGS "-O2 -std=c++17 -fPIC")
set(CMAKE_CXX_FLAGS "-O2 -std=c++17 -Wall -Wextra -fPIC -Wno-sign-compare")

# Find pybind11
set(PYBIND11_FINDPYTHON ON)
set(pybind11_DIR "/usr/local/lib/python3.10/dist-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# Add the hanabi library, compiling the core C++ files
add_library(hanabi STATIC
    hanabi_card.cc
    hanabi_game.cc
    hanabi_hand.cc
    hanabi_history_item.cc
    hanabi_move.cc
    hanabi_observation.cc
    hanabi_state.cc
    util.cc
    canonical_encoders.cc
)

# Add include directories for the hanabi library
target_include_directories(hanabi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Create Python module directly, without a separate pybind.cpp file
pybind11_add_module(hanabi_lib MODULE
    hanabi_card.cc
    hanabi_game.cc
    hanabi_hand.cc
    hanabi_history_item.cc
    hanabi_move.cc
    hanabi_observation.cc
    hanabi_state.cc
    util.cc
    canonical_encoders.cc
)

# Link the Python extension module to the hanabi library
target_link_libraries(hanabi_lib PRIVATE hanabi)

# Specify the output directory for the built Python module
set_target_properties(hanabi_lib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../build/python_bindings"
)
