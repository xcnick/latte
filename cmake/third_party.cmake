# These lists are later turned into target properties on main latte library target
set(External_PROJECT_TARGETS "")

# ---[ Threads
find_package(Threads REQUIRED)
list(APPEND Latte_LINKER_LIBS PRIVATE ${CMAKE_THREAD_LIBS_INIT})

# ---[ OpenMP
if(USE_OPENMP)
  # Ideally, this should be provided by the BLAS library IMPORTED target. However,
  # nobody does this, so we need to link to OpenMP explicitly and have the maintainer
  # to flick the switch manually as needed.
  #
  # Moreover, OpenMP package does not provide an IMPORTED target as well, and the
  # suggested way of linking to OpenMP is to append to CMAKE_{C,CXX}_FLAGS.
  # However, this naïve method will force any user of Latte to add the same kludge
  # into their buildsystem again, so we put these options into per-target PUBLIC
  # compile options and link flags, so that they will be exported properly.
  find_package(OpenMP REQUIRED)
  list(APPEND Latte_LINKER_LIBS PRIVATE ${OpenMP_CXX_FLAGS})
  list(APPEND Latte_COMPILE_OPTIONS PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# ---[ Google-protobuf
include(cmake/External/protobuf.cmake)
list(APPEND Latte_INCLUDE_DIRS PRIVATE ${PROTOBUF_INCLUDE_DIR})
list(APPEND Latte_LINKER_LIBS PRIVATE ${PROTOBUF_LIBRARIES})

# ---[ Google-glog
include(cmake/External/glog.cmake)
list(APPEND Latte_INCLUDE_DIRS PUBLIC ${GLOG_INCLUDE_DIRS})
list(APPEND Latte_LINKER_LIBS PUBLIC ${GLOG_LIBRARIES})

# ---[ Google-gflags
include(cmake/External/gflags.cmake)
list(APPEND Latte_INCLUDE_DIRS PRIVATE ${GFLAGS_INCLUDE_DIRS})
list(APPEND Latte_LINKER_LIBS PRIVATE ${GFLAGS_LIBRARIES})

# ---[ Google-gtest
include(cmake/External/gtest.cmake)
list(APPEND Latte_INCLUDE_DIRS PRIVATE ${GTEST_INCLUDE_DIRS})
list(APPEND Latte_LINKER_LIBS PRIVATE ${GTEST_LIBRARIES})

# ---[ HDF5
# find_package(HDF5 COMPONENTS HL REQUIRED)
# list(APPEND Latte_INCLUDE_DIRS PUBLIC ${HDF5_INCLUDE_DIRS})
# list(APPEND Latte_LINKER_LIBS PUBLIC ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

# This code is taken from https://github.com/sh1r0/latte-android-lib
# if(USE_HDF5)
#   find_package(HDF5 COMPONENTS HL REQUIRED)
#   include_directories(SYSTEM ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR})
#   list(APPEND Latte_LINKER_LIBS ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
#   add_definitions(-DUSE_HDF5)
# endif()

# ---[ LMDB
# if(USE_LMDB)
#   find_package(LMDB REQUIRED)
#   list(APPEND Latte_INCLUDE_DIRS PUBLIC ${LMDB_INCLUDE_DIR})
#   list(APPEND Latte_LINKER_LIBS PUBLIC ${LMDB_LIBRARIES})
#   list(APPEND Latte_DEFINITIONS PUBLIC -DUSE_LMDB)
#   if(ALLOW_LMDB_NOLOCK)
#     list(APPEND Latte_DEFINITIONS PRIVATE -DALLOW_LMDB_NOLOCK)
#   endif()
# endif()

# ---[ LevelDB
# if(USE_LEVELDB)
#   find_package(LevelDB REQUIRED)
#   list(APPEND Latte_INCLUDE_DIRS PUBLIC ${LevelDB_INCLUDES})
#   list(APPEND Latte_LINKER_LIBS PUBLIC ${LevelDB_LIBRARIES})
#   list(APPEND Latte_DEFINITIONS PUBLIC -DUSE_LEVELDB)
# endif()

# ---[ Snappy
# if(USE_LEVELDB)
#   find_package(Snappy REQUIRED)
#   list(APPEND Latte_INCLUDE_DIRS PRIVATE ${Snappy_INCLUDE_DIR})
#   list(APPEND Latte_LINKER_LIBS PRIVATE ${Snappy_LIBRARIES})
# endif()

# ---[ CUDA
include(cmake/cuda.cmake)

# if(NOT HAVE_CUDA)
#   if(CPU_ONLY)
#     message(STATUS "-- CUDA is disabled. Building without it...")
#   else()
#     message(WARNING "-- CUDA is not detected by cmake. Building without it...")
#   endif()

#   list(APPEND Latte_DEFINITIONS PUBLIC -DCPU_ONLY)
# endif()

# if(USE_NCCL)
#   find_package(NCCL REQUIRED)
#   include_directories(SYSTEM ${NCCL_INCLUDE_DIR})
#   list(APPEND Latte_LINKER_LIBS ${NCCL_LIBRARIES})
#   add_definitions(-DUSE_NCCL)
# endif()

# ---[ OpenCV
# if(USE_OPENCV)
#   find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
#   if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
#     find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
#   endif()
#   list(APPEND Latte_INCLUDE_DIRS PUBLIC ${OpenCV_INCLUDE_DIRS})
#   list(APPEND Latte_LINKER_LIBS PUBLIC ${OpenCV_LIBS})
#   message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
#   list(APPEND Latte_DEFINITIONS PUBLIC -DUSE_OPENCV)
# endif()

# ---[ BLAS
set(BLAS "Open" CACHE STRING "Selected BLAS library")
set_property(CACHE BLAS PROPERTY STRINGS "Atlas;Open;MKL")

if(BLAS STREQUAL "Open" OR BLAS STREQUAL "open")
  set(BLA_VENDOR OpenBLAS)
elseif(BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
  set(BLA_VENDOR MKL)
endif()

find_package(BLAS REQUIRED)
list(APPEND Latte_LINKER_LIBS PUBLIC ${BLAS_LIBRARIES})
