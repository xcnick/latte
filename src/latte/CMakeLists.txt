# generate protobuf sources
file(GLOB proto_files proto/*.proto)
latte_protobuf_generate_cpp(${proto_gen_folder} proto_srcs proto_hdrs ${proto_files})

# include python files either to force generation
add_library(latteproto STATIC ${proto_hdrs} ${proto_srcs})
# latte_default_properties(latteproto)
target_link_libraries(latteproto PUBLIC ${PROTOBUF_LIBRARIES})
target_include_directories(latteproto PUBLIC ${PROTOBUF_INCLUDE_DIR})

list(INSERT Latte_LINKER_LIBS 0 PUBLIC latteproto) # note, crucial to prepend!
add_dependencies(latteproto protobuf)

add_subdirectory(util)
add_subdirectory(layers)

# # --[ Latte library

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

#message("${Latte_INCLUDE_DIRS}")

target_include_directories(latte PRIVATE ${Latte_INCLUDE_DIRS})
target_link_libraries(latte PRIVATE ${Latte_LINKER_LIBS} latte_layers latte_util)


# # creates 'test_srcs', 'srcs', 'test_cuda', 'cuda' lists
# latte_pickup_latte_sources(${PROJECT_SOURCE_DIR})

# if(HAVE_CUDA)
#   latte_cuda_compile(cuda_objs ${cuda})
#   list(APPEND srcs ${cuda_objs} ${cuda})
# endif()

# add_library(latte ${srcs})
# target_link_libraries(latte ${Latte_LINKER_LIBS})
# latte_default_properties(latte)
# target_include_directories(latte ${Latte_INCLUDE_DIRS}
#                                  PUBLIC
#                                  $<BUILD_INTERFACE:${Latte_INCLUDE_DIR}>
#                                  $<INSTALL_INTERFACE:include>)
target_compile_definitions(latte PRIVATE ${Latte_DEFINITIONS})
if(Latte_COMPILE_OPTIONS)
  target_compile_options(latte PRIVATE ${Latte_COMPILE_OPTIONS})
endif()
# set_target_properties(latte PROPERTIES
#     VERSION   ${LATTE_TARGET_VERSION}
#     SOVERSION ${LATTE_TARGET_SOVERSION}
#     )

# # ---[ Tests
add_subdirectory(test)
