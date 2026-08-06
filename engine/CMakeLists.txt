cmake_minimum_required(VERSION 3.20)
project(cornerstone_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(CheckCXXCompilerFlag)

add_compile_options(-O3 -funroll-loops -Wall -Wextra)

# 开发机是 aarch64（Grace / Neoverse-V2），不要假设 x86 的 -march。
# 优先 -mcpu=native，退回 -march=native，都不支持就什么都不加。
check_cxx_compiler_flag("-mcpu=native" HAVE_MCPU_NATIVE)
if(HAVE_MCPU_NATIVE)
  add_compile_options(-mcpu=native)
else()
  check_cxx_compiler_flag("-march=native" HAVE_MARCH_NATIVE)
  if(HAVE_MARCH_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# 用容器里 python 侧自带的 pybind11，不额外拉源码
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYBIND11_LOOKUP)
if(NOT PYBIND11_LOOKUP EQUAL 0)
  message(FATAL_ERROR "找不到 pybind11，请在容器内 pip install pybind11")
endif()
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_engine
  src/pieces.cpp
  src/board.cpp
  src/playout.cpp
  src/agents.cpp
  src/reference.cpp
  src/bindings.cpp)

target_include_directories(_engine PRIVATE include)
find_package(Threads REQUIRED)
target_link_libraries(_engine PRIVATE Threads::Threads)

# 直接产出到 python 包目录，装不装包都能 import
set_target_properties(_engine PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../cornerstone")
