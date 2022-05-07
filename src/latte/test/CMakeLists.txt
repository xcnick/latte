enable_testing()

add_executable(test_latte)
target_sources(test_latte
  PRIVATE
    test_benchmark.cc
    test_blob.cc
    test_common.cc
    test_dummy_data_layer.cc
    #test_euclidean_loss_layer.cc
    test_filler.cc
    test_inner_product_layer.cc
    test_latte_main.cc
    test_layer_factory.cc
    #test_net.cc
    test_softmax_layer.cc
    test_syncedmem.cc
)

include(GoogleTest)
gtest_discover_tests(test_latte)

target_include_directories(test_latte PRIVATE ${Latte_INCLUDE_DIRS})
target_link_libraries(test_latte PRIVATE ${Latte_LINKER_LIBS} latte)

target_compile_definitions(test_latte PRIVATE ${Latte_DEFINITIONS})
if(Latte_COMPILE_OPTIONS)
  target_compile_options(test_latte PRIVATE ${Latte_COMPILE_OPTIONS})
endif()

add_dependencies(test_latte ${External_PROJECT_TARGETS})
