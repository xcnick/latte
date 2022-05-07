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

set(COMPILER_VERSION_ERROR_MSG "At least gcc 7, clang 5 or Apple clang 12 is supported.")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 7)
    message(FATAL_ERROR ${COMPILER_VERSION_ERROR_MSG})
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 5)
    message(FATAL_ERROR ${COMPILER_VERSION_ERROR_MSG})
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  if("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 12)
    message(FATAL_ERROR ${COMPILER_VERSION_ERROR_MSG})
  endif()
else()
  message(WARNING "Unknown compiler \"${CMAKE_CXX_COMPILER_ID}\".")
endif()

set(latte_cmake_dir ${PROJECT_SOURCE_DIR}/cmake)


# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/External)

# ---[ Options
option(BUILD_CUDA "" ON)
if (BUILD_CUDA)
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

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (BUILD_CUDA)
  # NOTE: if you want to use source PTX with a version different from produced PTX/binary, you should add flags
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    list(APPEND CMAKE_CUDA_ARCHITECTURES 60-real)
    # Tesla P40/P4, Quadro Pxxx/Pxxxx, GeForce GTX 10xx, TITAN X/Xp
    list(APPEND CMAKE_CUDA_ARCHITECTURES 61-real)
    # V100, TITAN V
    list(APPEND CMAKE_CUDA_ARCHITECTURES 70-real)
    if(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
      # T4, Quadro RTX xxxx, Txxxx, Geforce RTX 20xx, TITAN RTX
      list(APPEND CMAKE_CUDA_ARCHITECTURES 75-real)
    endif()
    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
      # A100
      list(APPEND CMAKE_CUDA_ARCHITECTURES 80-real)
    endif()
    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.1")
      # GeForce RTX 30xx
      list(APPEND CMAKE_CUDA_ARCHITECTURES 86-real)
    endif()
    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
      list(APPEND CMAKE_CUDA_ARCHITECTURES 80-virtual)
    elseif(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
      list(APPEND CMAKE_CUDA_ARCHITECTURES 75-virtual)
    else()
      list(APPEND CMAKE_CUDA_ARCHITECTURES 70-virtual)
    endif()
  endif()
  enable_language(CUDA)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
  set(CUDA_SEPARABLE_COMPILATION OFF)

  if ("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.2")
      set(CUDA_NVCC_THREADS_NUMBER "4" CACHE STRING "")
      list(APPEND CUDA_NVCC_FLAGS -t ${CUDA_NVCC_THREADS_NUMBER})
    endif()
    message(STATUS "CUDA_NVCC_FLAGS: " ${CUDA_NVCC_FLAGS})
    list(JOIN CUDA_NVCC_FLAGS " " CMAKE_CUDA_FLAGS)
  endif()
endif()

# ---[ Subdirectories
add_subdirectory(src/latte)
add_subdirectory(tools)
