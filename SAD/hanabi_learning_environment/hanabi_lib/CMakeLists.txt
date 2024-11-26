# Define the project
project(hanabi_lib)

# Specify the source files
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
    pybind.cc  # Add pybind.cc for Python bindings
)

# Create the library (shared object) for Python
add_library(hanabi SHARED ${SOURCE_FILES})

# Include directories
target_include_directories(hanabi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Find pybind11 and set flags for C++ compilation
set(PYBIND11_FINDPYTHON ON)
set(pybind11_DIR "/usr/local/lib/python3.10/dist-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)
target_link_libraries(hanabi PRIVATE pybind11::module)

# Set necessary compilation flags
set(CMAKE_CXX_FLAGS "-O2 -std=c++17 -Wall -Wextra -fPIC -Wno-sign-compare")