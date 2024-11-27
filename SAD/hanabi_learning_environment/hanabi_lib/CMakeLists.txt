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
)

# Create the library (shared object) for Python
add_library(hanabi SHARED ${SOURCE_FILES})

# Include directories
target_include_directories(hanabi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})