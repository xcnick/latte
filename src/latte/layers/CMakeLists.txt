# --[ Latte layers library
add_library(latte_layers STATIC)
target_sources(latte_layers
  PRIVATE
    dummy_data_layer.cc
    inner_product_layer.cc
    neuron_layer.cc
    sigmoid_layer.cc
    softmax_layer.cc
)

if (USE_CUDA)
  target_sources(latte_layers
    PRIVATE
      inner_product_layer.cu
      sigmoid_layer.cu
      softmax_layer.cu
  )
  if (USE_CUDNN)
    target_sources(latte_layers
      PRIVATE
        cudnn_sigmoid_layer.cc
        cudnn_softmax_layer.cc
    )
  endif()
endif()

target_include_directories(latte_layers PRIVATE ${Latte_INCLUDE_DIRS})
target_link_libraries(latte_layers PRIVATE ${Latte_LINKER_LIBS})

target_compile_definitions(latte_layers PRIVATE ${Latte_DEFINITIONS})
if(Latte_COMPILE_OPTIONS)
  target_compile_options(latte_layers PRIVATE ${Latte_COMPILE_OPTIONS})
endif()

add_dependencies(latte_layers ${External_PROJECT_TARGETS})
