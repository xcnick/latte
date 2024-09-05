cmake_minimum_required(VERSION 3.18.0)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE MATCHES "^(Debug|Release|RelWithDebInfo|MinSizeRel)$")
  message(FATAL_ERROR "Expected CMAKE_BUILD_TYPE is Debug, Release, RelWithDebInfo or MinSizeRel, got ${CMAKE_BUILD_TYPE}")
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# ---[ Latte project
project(Latte C CXX)

set(latte_cmake_dir ${PROJECT_SOURCE_DIR}/cmake)


# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/External)

# ---[ Options
option(USE_CUDA "" ON)
if (USE_CUDA)
  option(USE_CUDNN "Build Latte with cuDNN library support" ON)
endif()
option(USE_OPENMP "" ON)

set(Latte_INCLUDE_DIRS "")
set(Latte_LINKER_LIBS "")
set(Latte_DEFINITIONS "")
set(Latte_COMPILE_OPTIONS -Wall)

# ---[ Includes
list(APPEND Latte_INCLUDE_DIRS PUBLIC ${PROJECT_SOURCE_DIR}/include)

# ---[ third_party
set(THIRD_PARTY_PATH ${PROJECT_BINARY_DIR}/third_party)
include(cmake/third_party.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---[ Subdirectories
add_subdirectory(src/latte)
add_subdirectory(tools)
