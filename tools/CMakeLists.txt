add_executable(latte_main)
target_sources(latte_main
  PRIVATE
    latte.cc
)

target_include_directories(latte_main PRIVATE ${Latte_INCLUDE_DIRS})
target_link_libraries(latte_main PRIVATE ${Latte_LINKER_LIBS} latte)

target_compile_definitions(latte_main PRIVATE ${Latte_DEFINITIONS})
if(Latte_COMPILE_OPTIONS)
  target_compile_options(latte_main PRIVATE ${Latte_COMPILE_OPTIONS})
endif()

add_dependencies(latte_main ${External_PROJECT_TARGETS})
