# --[ Latte util library
add_library(latte_util STATIC)
target_sources(latte_util
  PRIVATE
    benchmark.cc
    insert_splits.cc
    io.cc
    math_functions.cc
    simple_logging.cc
)

if (USE_CUDA)
  target_sources(latte_util
    PRIVATE
      math_functions.cu
  )
endif()

target_include_directories(latte_util PRIVATE ${Latte_INCLUDE_DIRS})
target_link_libraries(latte_util PRIVATE ${Latte_LINKER_LIBS})

target_compile_definitions(latte_util PRIVATE ${Latte_DEFINITIONS})
if(Latte_COMPILE_OPTIONS)
  target_compile_options(latte_util PRIVATE ${Latte_COMPILE_OPTIONS})
endif()

add_dependencies(latte_util ${External_PROJECT_TARGETS})
