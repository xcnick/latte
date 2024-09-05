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

# ---[ CUDA
include(cmake/cuda.cmake)

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
