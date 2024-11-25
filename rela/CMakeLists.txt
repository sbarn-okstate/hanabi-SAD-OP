cmake_minimum_required(VERSION 3.12 FATAL_ERROR)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors")


# get and append paths for finding dep
execute_process(
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})


# find packages & third_party
find_package(Python REQUIRED COMPONENTS Interpreter Development)
link_directories(${Python_LIBRARIES})
find_package(Torch REQUIRED)


#add_subdirectory(
#  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/pybind11 third_party/pybind11
#)


# lib for other c++ programs
add_library(_rela
  types.cc
)
target_include_directories(_rela PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(_rela PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(_rela PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(_rela PUBLIC torch)


# python lib
pybind11_add_module(rela pybind.cc)
target_link_libraries(rela PUBLIC _rela)
