#project (hanabi_learning_environment)
project (pyhanabi)
set(CMAKE_C_FLAGS "-O2 -std=c++17 -fPIC")
set(CMAKE_CXX_FLAGS "-O2 -std=c++17 -Wall -Wextra -fPIC -Wno-sign-compare")

set(PYBIND11_FINDPYTHON ON)
set(pybind11_DIR "/usr/local/lib/python3.10/dist-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)
add_library(pyhanabi MODULE pybind.cpp)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hanabi_lib)

target_link_libraries(hanabi_lib PUBLIC pybind11::module)
target_include_directories(hanabi_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/hanabi_lib)

add_library (pyhanabi SHARED pyhanabi.cc)
target_link_libraries (pyhanabi PUBLIC hanabi)
