cmake_minimum_required(VERSION 3.24)
project(pmpp_cuda_study LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/examples/*/main.cu")
foreach(EXAMPLE_SOURCE IN LISTS EXAMPLE_SOURCES)
  get_filename_component(EXAMPLE_DIR "${EXAMPLE_SOURCE}" DIRECTORY)
  get_filename_component(EXAMPLE_NAME "${EXAMPLE_DIR}" NAME)
  add_executable(${EXAMPLE_NAME} "${EXAMPLE_SOURCE}")
  target_include_directories(${EXAMPLE_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
  set_target_properties(${EXAMPLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()
