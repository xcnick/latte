
################################################################################################
# Helper function to fetch latte includes which will be passed to dependent projects
# Usage:
#   latte_get_current_includes(<includes_list_variable>)
function(latte_get_current_includes includes_variable)
  get_property(current_includes DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
  latte_convert_absolute_paths(current_includes)

  # remove at most one ${PROJECT_BINARY_DIR} include added for latte_config.h
  list(FIND current_includes ${PROJECT_BINARY_DIR} __index)
  list(REMOVE_AT current_includes ${__index})

  # removing numpy includes (since not required for client libs)
  set(__toremove "")
  foreach(__i ${current_includes})
    if(${__i} MATCHES "python")
      list(APPEND __toremove ${__i})
    endif()
  endforeach()
  if(__toremove)
    list(REMOVE_ITEM current_includes ${__toremove})
  endif()

  latte_list_unique(current_includes)
  set(${includes_variable} ${current_includes} PARENT_SCOPE)
endfunction()

################################################################################################
# Helper function to get all list items that begin with given prefix
# Usage:
#   latte_get_items_with_prefix(<prefix> <list_variable> <output_variable>)
function(latte_get_items_with_prefix prefix list_variable output_variable)
  set(__result "")
  foreach(__e ${${list_variable}})
    if(__e MATCHES "^${prefix}.*")
      list(APPEND __result ${__e})
    endif()
  endforeach()
  set(${output_variable} ${__result} PARENT_SCOPE)
endfunction()

################################################################################################
# Function for generation Latte build- and install- tree export config files
# Usage:
#  latte_generate_export_configs()
function(latte_generate_export_configs)
  set(install_cmake_suffix "share/Latte")

  # ---[ Configure build-tree LatteConfig.cmake file ]---
  latte_get_current_includes(Latte_INCLUDE_DIRS)

  set(Latte_DEFINITIONS "")
  if(NOT HAVE_CUDA)
    set(HAVE_CUDA FALSE)
  endif()

  if(NOT HAVE_CUDNN)
    set(HAVE_CUDNN FALSE)
  else()
    list(APPEND DEFINITIONS -DUSE_CUDNN)
  endif()

  if(BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
    list(APPEND Latte_DEFINITIONS -DUSE_MKL)
  endif()

  if(NCCL_FOUND)
    list(APPEND Latte_DEFINITIONS -DUSE_NCCL)
  endif()

  if(TEST_FP16)
    list(APPEND Latte_DEFINITIONS -DTEST_FP16=1)
  endif()

  configure_file("cmake/Templates/LatteConfig.cmake.in" "${PROJECT_BINARY_DIR}/LatteConfig.cmake" @ONLY)

  # Add targets to the build-tree export set
  export(TARGETS latte latteproto FILE "${PROJECT_BINARY_DIR}/LatteTargets.cmake")
  export(PACKAGE Latte)

  # ---[ Configure install-tree LatteConfig.cmake file ]---

  # remove source and build dir includes
  latte_get_items_with_prefix(${PROJECT_SOURCE_DIR} Latte_INCLUDE_DIRS __insource)
  latte_get_items_with_prefix(${PROJECT_BINARY_DIR} Latte_INCLUDE_DIRS __inbinary)
  list(REMOVE_ITEM Latte_INCLUDE_DIRS ${__insource} ${__inbinary})

  # add `install` include folder
  set(lines
     "get_filename_component(__latte_include \"\${Latte_CMAKE_DIR}/../../include\" ABSOLUTE)\n"
     "list(APPEND Latte_INCLUDE_DIRS \${__latte_include})\n"
     "unset(__latte_include)\n")
  string(REPLACE ";" "" Latte_INSTALL_INCLUDE_DIR_APPEND_COMMAND ${lines})

  configure_file("cmake/Templates/LatteConfig.cmake.in" "${PROJECT_BINARY_DIR}/cmake/LatteConfig.cmake" @ONLY)

  # Install the LatteConfig.cmake and export set to use with install-tree
  # install(FILES "${PROJECT_BINARY_DIR}/cmake/LatteConfig.cmake" DESTINATION ${install_cmake_suffix})
  # install(EXPORT LatteTargets DESTINATION ${install_cmake_suffix})

  # ---[ Configure and install version file ]---

  # TODO: Lines below are commented because Latte does't declare its version in headers.
  # When the declarations are added, modify `latte_extract_latte_version()` macro and uncomment

  # configure_file(cmake/Templates/LatteConfigVersion.cmake.in "${PROJECT_BINARY_DIR}/LatteConfigVersion.cmake" @ONLY)
  # install(FILES "${PROJECT_BINARY_DIR}/LatteConfigVersion.cmake" DESTINATION ${install_cmake_suffix})
endfunction()
