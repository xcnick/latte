# include(FetchContent)
# FetchContent_Declare(
#   openblas
#   GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
#   GIT_TAG        v0.3.13
# )
#
# FetchContent_MakeAvailable(openblas)

include(ExternalProject)

set(OPENBLAS_INSTALL_DIR ${THIRD_PARTY_PATH}/openblas)
set(OPENBLAS_INCLUDE_DIRS ${THIRD_PARTY_PATH}/openblas/include/openblas)
set(OPENBLAS_LIBRARIES ${THIRD_PARTY_PATH}/openblas/lib/libopenblas.a pthread)

ExternalProject_Add(
    openblas
    GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
    GIT_TAG        v0.3.19
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${OPENBLAS_INSTALL_DIR}
    GIT_SHALLOW
)

list(APPEND External_PROJECT_TARGETS openblas)
