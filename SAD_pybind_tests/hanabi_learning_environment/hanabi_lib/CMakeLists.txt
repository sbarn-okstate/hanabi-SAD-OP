
cmake_minimum_required(VERSION 3.12)
project(HanabiBindings)
# Find pybind11 and set flags for C++ compilation
set(PYBIND11_FINDPYTHON ON)
set(pybind11_DIR "/usr/local/lib/python3.10/dist-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)
target_link_libraries(hanabi PRIVATE pybind11::module)

# Find pybind11
find_package(pybind11 REQUIRED)

# Specify source files for the core C++ code
set(SOURCE_FILES
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

# Specify the Python bindings file
set(PYBIND_SOURCE pybind.cc)

# Create the shared library (Python extension module)
add_library(hanabi MODULE ${SOURCE_FILES} ${PYBIND_SOURCE})

# Include directories for the project
target_include_directories(hanabi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link pybind11 and Python to the library
target_link_libraries(hanabi PRIVATE pybind11::module)

# Set compilation flags
set(CMAKE_CXX_FLAGS "-O2 -std=c++17 -Wall -Wextra -fPIC")