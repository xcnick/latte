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

# ---[ Latte version
# set(LATTE_TARGET_VERSION "1.0.0" CACHE STRING "Latte logical version")
# set(LATTE_TARGET_SOVERSION "1.0.0" CACHE STRING "Latte soname version")
# add_definitions(-DLATTE_VERSION=${LATTE_TARGET_VERSION})

# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/External)

include(ExternalProject)
# include(GNUInstallDirs)

# include(cmake/Utils.cmake)
# include(cmake/Targets.cmake)
# include(cmake/Misc.cmake)
# include(cmake/Summary.cmake)
# include(cmake/ConfigGen.cmake)

# ---[ Options

option(BUILD_CUDA "" ON)
if (BUILD_CUDA)
  option(USE_CUDNN "Build Latte with cuDNN library support" ON)
endif()
option(USE_OPENMP "" ON)
#latte_option(CPU_ONLY  "Build Latte without CUDA support" OFF) # TODO: rename to USE_CUDA
#latte_option(USE_CUDNN "Build Latte with cuDNN library support" ON IF NOT CPU_ONLY)
# latte_option(USE_NCCL "Build Latte with NCCL library support" OFF)
# latte_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
# latte_option(BUILD_matlab "Build Matlab wrapper" OFF IF UNIX OR APPLE)
# latte_option(BUILD_docs   "Build documentation" ON IF UNIX OR APPLE)
# latte_option(BUILD_python_layer "Build the Latte Python layer" ON)
# latte_option(USE_OPENCV "Build with OpenCV support" ON)
# latte_option(USE_LEVELDB "Build with levelDB" ON)
# latte_option(USE_LMDB "Build with lmdb" ON)
# latte_option(ALLOW_LMDB_NOLOCK "Allow MDB_NOLOCK when reading LMDB files (only if necessary)" OFF)
# latte_option(USE_OPENMP "Link with OpenMP (when your BLAS wants OpenMP and you get linker errors)" OFF)

# This code is taken from https://github.com/sh1r0/latte-android-lib
# latte_option(USE_HDF5 "Build with hdf5" ON)

set(Latte_INCLUDE_DIRS "")
set(Latte_LINKER_LIBS "")
set(Latte_DEFINITIONS "")
set(Latte_COMPILE_OPTIONS "")

# ---[ Includes
list(APPEND Latte_INCLUDE_DIRS PUBLIC ${PROJECT_SOURCE_DIR}/include)

# ---[ third_party
set(THIRD_PARTY_PATH ${PROJECT_SOURCE_DIR}/third_party)
include(cmake/third_party.cmake)

# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

#latte_set_latte_link()

# if(USE_libstdcpp)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
#   message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
# endif()

# ---[ Warnings
#latte_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# ---[ Config generation
#configure_file(cmake/Templates/latte_config.h.in "${PROJECT_BINARY_DIR}/latte_config.h")

#set(Latte_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
#set(Latte_SRC_DIR ${PROJECT_SOURCE_DIR}/src)

#include_directories(${Latte_INCLUDE_DIR} ${PROJECT_BINARY_DIR} ${THIRD_PARTY_PATH})

# ---[ Includes & defines for CUDA

# cuda_compile() does not have per-call dependencies or include pathes
# (cuda_compile() has per-call flags, but we set them here too for clarity)
#
# list(REMOVE_ITEM ...) invocations remove PRIVATE and PUBLIC keywords from collected definitions and include pathes
# if(HAVE_CUDA)
#   # pass include pathes to cuda_include_directories()
#   set(Latte_ALL_INCLUDE_DIRS ${Latte_INCLUDE_DIRS})
#   list(REMOVE_ITEM Latte_ALL_INCLUDE_DIRS PRIVATE PUBLIC)
#   cuda_include_directories(${Latte_INCLUDE_DIR} ${Latte_SRC_DIR} ${Latte_ALL_INCLUDE_DIRS})

#   # add definitions to nvcc flags directly
#   set(Latte_ALL_DEFINITIONS ${Latte_DEFINITIONS})
#   list(REMOVE_ITEM Latte_ALL_DEFINITIONS PRIVATE PUBLIC)
#   list(APPEND CUDA_NVCC_FLAGS ${Latte_ALL_DEFINITIONS})
# endif()


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
#add_subdirectory(tools)
# add_subdirectory(examples)
# add_subdirectory(python)
# add_subdirectory(matlab)
# add_subdirectory(docs)


# ---[ Linter target
# add_custom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

# ---[ pytest target
# if(BUILD_python)
#   add_custom_target(pytest COMMAND python${python_version} -m unittest discover -s latte/test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python )
#   add_dependencies(pytest pylatte)
# endif()

# ---[ uninstall target
# configure_file(
#     ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Uninstall.cmake.in
#     ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake
#     IMMEDIATE @ONLY)

# add_custom_target(uninstall
#     COMMAND ${CMAKE_COMMAND} -P
#     ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake)

# ---[ Configuration summary
#latte_print_configuration_summary()

# ---[ Export configs generation
#latte_generate_export_configs()
