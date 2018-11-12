# glog depends on gflags
include("cmake/External/gflags.cmake")

if (NOT __GLOG_INCLUDED)
  set(__GLOG_INCLUDED TRUE)

  # try the system-wide glog first
  find_package(Glog)
  if (GLOG_FOUND)
      set(GLOG_EXTERNAL FALSE)
  else()
    # fetch and build glog from github

    # build directory
    set(glog_PREFIX ${CMAKE_BINARY_DIR}/external/glog-prefix)
    # install directory
    set(glog_INSTALL ${CMAKE_BINARY_DIR}/external/glog-install)

    # we build glog statically, but want to link it into the latte shared library
    # this requires position-independent code
    if (UNIX)
      set(GLOG_EXTRA_COMPILER_FLAGS "-fPIC")
    endif()

    set(GLOG_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${GLOG_EXTRA_COMPILER_FLAGS})
    set(GLOG_C_FLAGS ${CMAKE_C_FLAGS} ${GLOG_EXTRA_COMPILER_FLAGS})

    # depend on gflags if we're also building it
    if (GFLAGS_EXTERNAL)
      set(GLOG_DEPENDS gflags)
    endif()

    set(GLOG_FOUND TRUE)
    set(GLOG_INCLUDE_DIRS ${glog_INSTALL}/include)
    set(GLOG_LIBRARIES ${GFLAGS_LIBRARIES} ${glog_INSTALL}/lib/libglog.a)
    set(GLOG_LIBRARY_DIRS ${glog_INSTALL}/lib)
    set(GLOG_EXTERNAL TRUE)

    list(APPEND external_project_dependencies glog)
  endif()

endif()
