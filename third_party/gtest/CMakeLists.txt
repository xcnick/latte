add_library(gtest STATIC EXCLUDE_FROM_ALL gtest.h gtest-all.cpp)
latte_default_properties(gtest)
target_include_directories(gtest PUBLIC ${Latte_SRC_DIR})
#target_compile_definitions(gtest PUBLIC -DGTEST_USE_OWN_TR1_TUPLE)
