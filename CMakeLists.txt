cmake_minimum_required(VERSION 3.0.0)

# ---[ Latte project
project(Latte C CXX)

# ---[ Latte version
set(LATTE_TARGET_VERSION "1.0.0" CACHE STRING "Latte logical version")
set(LATTE_TARGET_SOVERSION "1.0.0" CACHE STRING "Latte soname version")
add_definitions(-DLATTE_VERSION=${LATTE_TARGET_VERSION})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release CACHE STRING "set build type to release")
set(CMAKE_CONFIGURATION_TYPES  Release CACHE STRING "set build type to release" FORCE)

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(ExternalProject)
# include(GNUInstallDirs)

include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/Misc.cmake)
include(cmake/Summary.cmake)
include(cmake/ConfigGen.cmake)

# ---[ Options
latte_option(CPU_ONLY  "Build Latte without CUDA support" OFF) # TODO: rename to USE_CUDA
latte_option(USE_CUDNN "Build Latte with cuDNN library support" ON IF NOT CPU_ONLY)
# latte_option(USE_NCCL "Build Latte with NCCL library support" OFF)
# latte_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
# latte_option(BUILD_matlab "Build Matlab wrapper" OFF IF UNIX OR APPLE)
# latte_option(BUILD_docs   "Build documentation" ON IF UNIX OR APPLE)
# latte_option(BUILD_python_layer "Build the Latte Python layer" ON)
latte_option(USE_OPENCV "Build with OpenCV support" ON)
# latte_option(USE_LEVELDB "Build with levelDB" ON)
# latte_option(USE_LMDB "Build with lmdb" ON)
# latte_option(ALLOW_LMDB_NOLOCK "Allow MDB_NOLOCK when reading LMDB files (only if necessary)" OFF)
# latte_option(USE_OPENMP "Link with OpenMP (when your BLAS wants OpenMP and you get linker errors)" OFF)

# This code is taken from https://github.com/sh1r0/latte-android-lib
# latte_option(USE_HDF5 "Build with hdf5" ON)

# ---[ Dependencies
include(cmake/Dependencies.cmake)

# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

latte_set_latte_link()

# if(USE_libstdcpp)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
#   message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
# endif()

# ---[ Warnings
latte_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# ---[ Config generation
configure_file(cmake/Templates/latte_config.h.in "${PROJECT_BINARY_DIR}/latte_config.h")

# ---[ Includes
set(Latte_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(Latte_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(THIRDPARTY_DIR ${PROJECT_SOURCE_DIR}/third_party)
include_directories(${Latte_INCLUDE_DIR} ${PROJECT_BINARY_DIR} ${THIRDPARTY_DIR})

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

# ---[ Subdirectories
add_subdirectory(${THIRDPARTY_DIR}/gtest)
#add_subdirectory(src/gtest)
add_subdirectory(src/latte)
add_subdirectory(tools)
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
latte_print_configuration_summary()

# ---[ Export configs generation
latte_generate_export_configs()
