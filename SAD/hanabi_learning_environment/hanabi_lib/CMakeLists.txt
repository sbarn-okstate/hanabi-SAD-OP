add_library (hanabi hanabi_card.cc hanabi_game.cc hanabi_hand.cc hanabi_history_item.cc hanabi_move.cc hanabi_observation.cc hanabi_state.cc util.cc canonical_encoders.cc)

# Find pybind11
set(PYBIND11_FINDPYTHON ON)
set(pybind11_DIR "/usr/local/lib/python3.10/dist-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# Add the pybind.cpp source file here
add_library(hanabi_lib MODULE 
    pybind.cpp
    canonical_encoders.cc
    # Add other necessary C++ source files here
)

# Link with pybind11
target_link_libraries(hanabi_lib PRIVATE pybind11::module)
target_include_directories(hanabi_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(hanabi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})