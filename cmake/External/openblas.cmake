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

ExternalProject_add(
    extern_openblas
    GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
    GIT_TAG        v0.3.13
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${OPENBLAS_INSTALL_DIR}
    GIT_SHALLOW
)

add_library(openblas STATIC IMPORTED GLOBAL)
set_property(TARGET openblas PROPERTY IMPORTED_LOCATION ${OPENBLAS_LIBRARIES})
add_dependencies(openblas extern_openblas)
#link_libraries(gtest)