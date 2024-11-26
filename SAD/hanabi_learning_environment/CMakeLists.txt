project (hanabi_learning_environment)
set(CMAKE_C_FLAGS "-O2 -std=c++17 -fPIC")
set(CMAKE_CXX_FLAGS "-O2 -std=c++17 -Wall -Wextra -fPIC -Wno-sign-compare")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hanabi_lib)

add_library (pyhanabi SHARED pyhanabi.cc)

set(PYBIND11_FINDPYTHON ON)
set(pybind11_DIR "/usr/local/lib/python3.10/dist-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

pybind11_add_module(hanabi_lib MODULE
    pybind.cpp
    # Add other necessary source files here
)

target_link_libraries (pyhanabi PUBLIC hanabi)
target_link_libraries (hanabi_lib PUBLIC hanabi)