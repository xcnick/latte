# include(FetchContent)
# FetchContent_Declare(
#   gflags
#   GIT_REPOSITORY https://github.com/gflags/gflags.git
#   GIT_TAG        v2.2.2
# )
#
# FetchContent_MakeAvailable(gflags)

include(ExternalProject)

set(GFLAGS_INSTALL_DIR ${THIRD_PARTY_PATH}/gflags)
set(GFLAGS_INCLUDE_DIRS ${THIRD_PARTY_PATH}/gflags/include)
set(GFLAGS_LIBRARIES ${THIRD_PARTY_PATH}/gflags/lib/libgflags.a)

ExternalProject_Add(
    gflags
    GIT_REPOSITORY https://github.com/gflags/gflags.git
    GIT_TAG        v2.2.2
    GIT_SHALLOW
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GFLAGS_INSTALL_DIR} \\
               -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    BUILD_BYPRODUCTS ${GFLAGS_LIBRARIES}
)

list(APPEND External_PROJECT_TARGETS gflags)
