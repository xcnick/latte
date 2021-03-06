# include(FetchContent)
# FetchContent_Declare(
#   glog
#   GIT_REPOSITORY https://github.com/google/glog
#   GIT_TAG        v0.4.0
# )
#
# FetchContent_MakeAvailable(glog)

include(ExternalProject)

set(GLOG_INSTALL_DIR ${THIRD_PARTY_PATH}/glog)
set(GLOG_INCLUDE_DIRS ${THIRD_PARTY_PATH}/glog/include)
set(GLOG_LIBRARIES ${THIRD_PARTY_PATH}/glog/lib/libglog.a)

ExternalProject_add(
    glog
    GIT_REPOSITORY https://github.com/google/glog.git
    GIT_TAG        v0.4.0
    GIT_SHALLOW
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GLOG_INSTALL_DIR}
    #INSTALL_DIR ${GLOG_INSTALL_DIR}
)
