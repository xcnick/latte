# include(FetchContent)
# FetchContent_Declare(
#   protobuf
#   GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
#   GIT_TAG        v3.12.0
#   SOURCE_SUBDIR  cmake
# )
# set(protobuf_BUILD_TESTS OFF)
# FetchContent_MakeAvailable(protobuf)

include(ExternalProject)

set(PROTOBUF_INSTALL_DIR ${THIRD_PARTY_PATH}/protobuf)
set(PROTOBUF_INCLUDE_DIR ${THIRD_PARTY_PATH}/protobuf/include)
set(PROTOBUF_LIBRARIES ${THIRD_PARTY_PATH}/protobuf/lib/libprotobuf.a)

ExternalProject_Add(
    protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG        v3.20.1
    GIT_SHALLOW
    SOURCE_SUBDIR  cmake
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${PROTOBUF_INSTALL_DIR} \\
               -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER} \\
               -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER} \\
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \\
               -Dprotobuf_BUILD_EXAMPLES=OFF \\
               -Dprotobuf_BUILD_TESTS=OFF \\
               -DBUILD_SHARED_LIBS=OFF \\
               -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    BUILD_BYPRODUCTS ${PROTOBUF_LIBRARIES}
)

add_executable(protoc IMPORTED GLOBAL)
add_dependencies(protoc protobuf)
set_target_properties(
    protoc PROPERTIES
    IMPORTED_LOCATION ${PROTOBUF_INSTALL_DIR}/bin/protoc
)

#set(PROTOBUF_PROTOC_EXECUTABLE ${PROTOBUF_INSTALL_DIR}/bin/protoc)
#set(protobuf_MODULE_COMPATIBLE ON CACHE BOOL "")


#if(EXISTS ${PROTOBUF_PROTOC_EXECUTABLE})
#  message(STATUS "Found PROTOBUF Compiler: ${PROTOBUF_PROTOC_EXECUTABLE}")
#else()
#  message(FATAL_ERROR "Could not find PROTOBUF Compiler")
#endif()

# if(PROTOBUF_FOUND)
#   # fetches protobuf version
#   latte_parse_header(${PROTOBUF_INCLUDE_DIR}/google/protobuf/stubs/common.h VERION_LINE GOOGLE_PROTOBUF_VERSION)
#   string(REGEX MATCH "([0-9])00([0-9])00([0-9])" PROTOBUF_VERSION ${GOOGLE_PROTOBUF_VERSION})
#   set(PROTOBUF_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
#   unset(GOOGLE_PROTOBUF_VERSION)
# endif()

# place where to generate protobuf sources
set(proto_gen_folder ${PROJECT_BINARY_DIR}/include/latte/proto)
list(APPEND Latte_INCLUDE_DIRS PUBLIC ${PROJECT_BINARY_DIR}/include)

list(APPEND External_PROJECT_TARGETS protobuf)

set(PROTOBUF_GENERATE_CPP_APPEND_PATH TRUE)

# ################################################################################################
# # Modification of standard 'protobuf_generate_cpp()' with output dir parameter and python support
# # Usage:
# #   latte_protobuf_generate_cpp(<output_dir> <srcs_var> <hdrs_var> <proto_files>)
function(latte_protobuf_generate_cpp output_dir srcs_var hdrs_var)
  if(NOT ARGN)
    message(SEND_ERROR "Error: latte_protobuf_generate_cpp_py() called without any proto files")
    return()
  endif()

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(fil ${ARGN})
      get_filename_component(abs_fil ${fil} ABSOLUTE)
      get_filename_component(abs_path ${abs_fil} PATH)
      list(FIND _protoc_include ${abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protoc_include -I ${abs_path})
      endif()
    endforeach()
  else()
    set(_protoc_include -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  set(${srcs_var})
  set(${hdrs_var})
  foreach(fil ${ARGN})
    get_filename_component(abs_fil ${fil} ABSOLUTE)
    get_filename_component(fil_we ${fil} NAME_WE)

    list(APPEND ${srcs_var} "${output_dir}/${fil_we}.pb.cc")
    list(APPEND ${hdrs_var} "${output_dir}/${fil_we}.pb.h")

    add_custom_command(
      OUTPUT "${output_dir}/${fil_we}.pb.cc"
             "${output_dir}/${fil_we}.pb.h"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${output_dir}"
      COMMAND protoc --cpp_out ${output_dir} ${_protoc_include} ${abs_fil}
      DEPENDS ${abs_fil}
      COMMENT "Running C++ protocol buffer compiler on ${fil}" VERBATIM )
  endforeach()

  set_source_files_properties(${${srcs_var}} ${${hdrs_var}} PROPERTIES GENERATED TRUE)
  set(${srcs_var} ${${srcs_var}} PARENT_SCOPE)
  set(${hdrs_var} ${${hdrs_var}} PARENT_SCOPE)
endfunction()
