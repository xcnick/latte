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

ExternalProject_Add(
    glog
    GIT_REPOSITORY https://github.com/google/glog.git
    GIT_TAG        v0.5.0
    GIT_SHALLOW
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GLOG_INSTALL_DIR} \\
               -DBUILD_SHARED_LIBS=OFF \\
               -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)

ExternalProject_Add_StepDependencies(glog build gflags gtest)

list(APPEND External_PROJECT_TARGETS glog)
