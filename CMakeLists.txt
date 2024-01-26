cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set_property(GLOBAL PROPERTY USE_FOLDERS On)

message(STATUS "CMake version is ${CMAKE_VERSION}")

project(micrograd)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/tools/cmake" ${CMAKE_MODULE_PATH})
include(helper-functions)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "This project only supports 64-bit builds!")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CONFIGURATION_TYPES Debug Release MinSizeRel RelWithDebInfo)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
set(CMAKE_INSTALL_LOCAL_ONLY TRUE)

set(CPM_DOWNLOAD_VERSION 0.38.7)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_MODULE_PATH}/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT(EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake")
  file(DOWNLOAD https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()

include(${CPM_DOWNLOAD_LOCATION})

add_subdirectory(source)

option(BUILD_TESTS "Build the tests" ON)

if(BUILD_TESTS)
  enable_testing()
  include(CTest)
  include(GoogleTest)
  add_subdirectory(tests)
endif()