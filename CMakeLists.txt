cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(libtorch_win_test VERSION 0.1 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Torch QUIET PATHS "${CMAKE_SOURCE_DIR}/libtorch")

set(EXECUTABLE_NAME wintorch)

add_executable(${EXECUTABLE_NAME})

target_sources(${EXECUTABLE_NAME} PRIVATE wintorch.cpp)

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

target_link_libraries(${EXECUTABLE_NAME} "${TORCH_LIBRARIES}")

if(MSVC)
    include(copy_torch_dlls)
    copy_torch_dlls(${EXECUTABLE_NAME})
endif(MSVC)