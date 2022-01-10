# generate protobuf sources
file(GLOB proto_files proto/*.proto)
latte_protobuf_generate_cpp(${proto_gen_folder} proto_srcs proto_hdrs ${proto_files})

# include python files either to force generation
add_library(latteproto STATIC ${proto_hdrs} ${proto_srcs})
target_link_libraries(latteproto PUBLIC ${PROTOBUF_LIBRARIES})
target_include_directories(latteproto PUBLIC ${PROTOBUF_INCLUDE_DIR})

list(INSERT Latte_LINKER_LIBS 0 PUBLIC latteproto) # note, crucial to prepend!
add_dependencies(latteproto protobuf)

add_subdirectory(util)
add_subdirectory(layers)

# --[ Latte library
add_library(latte STATIC)
target_sources(latte
  PRIVATE
    blob.cc
    common.cc
    layer_factory.cc
    layer.cc
    net.cc
    syncedmem.cc
)

target_include_directories(latte PRIVATE ${Latte_INCLUDE_DIRS})
target_link_libraries(latte PRIVATE ${Latte_LINKER_LIBS} latte_layers latte_util)

target_compile_definitions(latte PRIVATE ${Latte_DEFINITIONS})
if(Latte_COMPILE_OPTIONS)
  target_compile_options(latte PRIVATE ${Latte_COMPILE_OPTIONS})
endif()

# ---[ Tests
add_subdirectory(test)
