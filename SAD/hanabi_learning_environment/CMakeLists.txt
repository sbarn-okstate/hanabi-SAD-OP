project (hanabi_learning_environment)

set(CMAKE_C_FLAGS "-O2 -std=c++17 -fPIC")
set(CMAKE_CXX_FLAGS "-O2 -std=c++17 -Wall -Wextra -fPIC -Wno-sign-compare")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hanabi_lib)

add_library (pyhanabi SHARED pyhanabi.cc)
target_link_libraries (pyhanabi PUBLIC hanabi)
