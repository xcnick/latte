################################################################################################
# Latte status report function.
# Automatically align right column and selects text based on condition.
# Usage:
#   latte_status(<text>)
#   latte_status(<heading> <value1> [<value2> ...])
#   latte_status(<heading> <condition> THEN <text for TRUE> ELSE <text for FALSE> )
function(latte_status text)
  set(status_cond)
  set(status_then)
  set(status_else)

  set(status_current_name "cond")
  foreach(arg ${ARGN})
    if(arg STREQUAL "THEN")
      set(status_current_name "then")
    elseif(arg STREQUAL "ELSE")
      set(status_current_name "else")
    else()
      list(APPEND status_${status_current_name} ${arg})
    endif()
  endforeach()

  if(DEFINED status_cond)
    set(status_placeholder_length 23)
    string(RANDOM LENGTH ${status_placeholder_length} ALPHABET " " status_placeholder)
    string(LENGTH "${text}" status_text_length)
    if(status_text_length LESS status_placeholder_length)
      string(SUBSTRING "${text}${status_placeholder}" 0 ${status_placeholder_length} status_text)
    elseif(DEFINED status_then OR DEFINED status_else)
      message(STATUS "${text}")
      set(status_text "${status_placeholder}")
    else()
      set(status_text "${text}")
    endif()

    if(DEFINED status_then OR DEFINED status_else)
      if(${status_cond})
        string(REPLACE ";" " " status_then "${status_then}")
        string(REGEX REPLACE "^[ \t]+" "" status_then "${status_then}")
        message(STATUS "${status_text} ${status_then}")
      else()
        string(REPLACE ";" " " status_else "${status_else}")
        string(REGEX REPLACE "^[ \t]+" "" status_else "${status_else}")
        message(STATUS "${status_text} ${status_else}")
      endif()
    else()
      string(REPLACE ";" " " status_cond "${status_cond}")
      string(REGEX REPLACE "^[ \t]+" "" status_cond "${status_cond}")
      message(STATUS "${status_text} ${status_cond}")
    endif()
  else()
    message(STATUS "${text}")
  endif()
endfunction()


################################################################################################
# Function for fetching Latte version from git and headers
# Usage:
#   latte_extract_latte_version()
function(latte_extract_latte_version)
  set(Latte_GIT_VERSION "unknown")
  find_package(Git)
  if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
                    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                    OUTPUT_VARIABLE Latte_GIT_VERSION
                    RESULT_VARIABLE __git_result)
    if(NOT ${__git_result} EQUAL 0)
      set(Latte_GIT_VERSION "unknown")
    endif()
  endif()

  set(Latte_GIT_VERSION ${Latte_GIT_VERSION} PARENT_SCOPE)
  set(Latte_VERSION "<TODO> (Latte doesn't declare its version in headers)" PARENT_SCOPE)

  # latte_parse_header(${Latte_INCLUDE_DIR}/latte/version.hpp Latte_VERSION_LINES LATTE_MAJOR LATTE_MINOR LATTE_PATCH)
  # set(Latte_VERSION "${LATTE_MAJOR}.${LATTE_MINOR}.${LATTE_PATCH}" PARENT_SCOPE)

  # or for #define Latte_VERSION "x.x.x"
  # latte_parse_header_single_define(Latte ${Latte_INCLUDE_DIR}/latte/version.hpp Latte_VERSION)
  # set(Latte_VERSION ${Latte_VERSION_STRING} PARENT_SCOPE)

endfunction()


################################################################################################
# Prints accumulated latte configuration summary
# Usage:
#   latte_print_configuration_summary()

function(latte_print_configuration_summary)
  latte_extract_latte_version()
  set(Latte_VERSION ${Latte_VERSION} PARENT_SCOPE)

  latte_merge_flag_lists(__flags_rel CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS)
  latte_merge_flag_lists(__flags_deb CMAKE_CXX_FLAGS_DEBUG   CMAKE_CXX_FLAGS)

  latte_status("")
  latte_status("******************* Latte Configuration Summary *******************")
  latte_status("General:")
  latte_status("  Version           :   ${LATTE_TARGET_VERSION}")
  latte_status("  Git               :   ${Latte_GIT_VERSION}")
  latte_status("  System            :   ${CMAKE_SYSTEM_NAME}")
  latte_status("  C++ compiler      :   ${CMAKE_CXX_COMPILER}")
  latte_status("  Release CXX flags :   ${__flags_rel}")
  latte_status("  Debug CXX flags   :   ${__flags_deb}")
  latte_status("  Build type        :   ${CMAKE_BUILD_TYPE}")
  latte_status("")
  latte_status("  BUILD_SHARED_LIBS :   ${BUILD_SHARED_LIBS}")
  latte_status("  CPU_ONLY          :   ${CPU_ONLY}")
  # latte_status("  USE_OPENCV        :   ${USE_OPENCV}")
  latte_status("")
  latte_status("Dependencies:")
  latte_status("  BLAS              : " APPLE THEN "Yes (vecLib)" ELSE "Yes (${BLAS})")
  latte_status("  glog              :   Yes")
  latte_status("  gflags            :   Yes")
  latte_status("  protobuf          : " PROTOBUF_FOUND THEN "Yes (ver. ${PROTOBUF_VERSION})" ELSE "No" )
  # if(USE_OPENCV)
  #   latte_status("  OpenCV            :   Yes (ver. ${OpenCV_VERSION})")
  # endif()
  latte_status("  CUDA              : " HAVE_CUDA THEN "Yes (ver. ${CUDA_VERSION})" ELSE "No" )
  latte_status("")
  if(HAVE_CUDA)
    latte_status("NVIDIA CUDA:")
    latte_status("  Target GPU(s)     :   ${CUDA_ARCH_NAME}" )
    latte_status("  GPU arch(s)       :   ${NVCC_FLAGS_EXTRA_readable}")
    if(USE_CUDNN)
      latte_status("  cuDNN             : " HAVE_CUDNN THEN "Yes (ver. ${CUDNN_VERSION})" ELSE "Not found")
    else()
      latte_status("  cuDNN             :   Disabled")
    endif()
    latte_status("")
  endif()

  latte_status("Install:")
  latte_status("  Install path      :   ${CMAKE_INSTALL_PREFIX}")
  latte_status("")
endfunction()
