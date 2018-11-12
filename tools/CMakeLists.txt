# Collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

# Build each source file independently
foreach(source ${srcs})
  get_filename_component(name ${source} NAME_WE)

  # latte target already exits
  if(name MATCHES "latte")
    set(name ${name}.bin)
  endif()

  # target
  add_executable(${name} ${source})
  target_link_libraries(${name} ${Latte_LINK})
  latte_default_properties(${name})

  # set back RUNTIME_OUTPUT_DIRECTORY
  latte_set_runtime_directory(${name} "${PROJECT_BINARY_DIR}/tools")
  latte_set_solution_folder(${name} tools)

  # restore output name without suffix
  if(name MATCHES "latte.bin")
    set_target_properties(${name} PROPERTIES OUTPUT_NAME latte)
  endif()

endforeach(source)
