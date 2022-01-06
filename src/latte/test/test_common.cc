
#include "latte/common.h"
#include "latte/test/test_latte_main.h"

namespace latte {

class CommonTest : public ::testing::Test {};

#ifdef WITH_CUDA

TEST_F(CommonTest, TestCublasHandlerGPU) {
  int cuda_device_id;
  CUDA_CHECK(cudaGetDevice(&cuda_device_id));
  EXPECT_TRUE(Latte::cublas_handle());
}

#endif

TEST_F(CommonTest, TestBrewMode) {
  Latte::set_mode(Latte::CPU);
  EXPECT_EQ(Latte::mode(), Latte::CPU);
  Latte::set_mode(Latte::GPU);
  EXPECT_EQ(Latte::mode(), Latte::GPU);
}

}  // namespace latte
