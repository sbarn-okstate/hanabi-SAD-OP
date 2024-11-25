cmake_minimum_required (VERSION 3.3)
project (hanalearn)

set(CMAKE_C_FLAGS "-O2 -std=c++11 -fPIC")
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -O2 -std=c++11 -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors")

add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/rela)
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/hanabi_learning_environment/hanabi_lib)
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/hanabi_learning_environment)


pybind11_add_module(hanalearn ${CMAKE_CURRENT_SOURCE_DIR}/cpp/pybind.cc)
target_link_libraries(hanalearn PUBLIC hanabi)
target_link_libraries(hanalearn PUBLIC _rela)
target_include_directories(hanalearn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})