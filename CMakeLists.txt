CMAKE_MINIMUM_REQUIRED(VERSION 3.3)
project(hanabi-SAD-OP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hanabi_learning_environment)
