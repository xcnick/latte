# include(FetchContent)
# FetchContent_Declare(
#   gtest
#   GIT_REPOSITORY https://github.com/google/googletest.git
#   GIT_TAG        release-1.8.0
# )
#
# FetchContent_MakeAvailable(gtest)

include(ExternalProject)

set(GTEST_INSTALL_DIR ${THIRD_PARTY_PATH}/gtest)
set(GTEST_INCLUDE_DIRS ${THIRD_PARTY_PATH}/gtest/include)
set(GTEST_LIBRARIES ${THIRD_PARTY_PATH}/gtest/lib/libgtest.a)

ExternalProject_add(
    gtest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.11.0
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL_DIR}
    GIT_SHALLOW
)
