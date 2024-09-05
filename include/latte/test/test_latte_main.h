// The main latte test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#ifndef LATTE_TEST_TEST_LATTE_MAIN_H_
#define LATTE_TEST_TEST_LATTE_MAIN_H_

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include "latte/common.h"

using std::cout;
using std::endl;

#define CUDA_TEST_DEVICE -1

namespace latte {

template <typename TypeParam>
class MultiDeviceTest : public ::testing::Test {
 public:
  using Dtype = typename TypeParam::Dtype;

 protected:
  MultiDeviceTest() { Latte::set_mode(TypeParam::device); }
  virtual ~MultiDeviceTest() {}
};

using TestDtypes = ::testing::Types<float, double>;

template <typename TypeParam>
struct CPUDevice {
  using Dtype = TypeParam;
  static const Latte::Brew device = Latte::CPU;
};

template <typename Dtype>
class CPUDeviceTest : public MultiDeviceTest<CPUDevice<Dtype>> {};

#ifndef USE_CUDA

using TestDtypesAndDevices =
    ::testing::Types<CPUDevice<float>, CPUDevice<double>>;

#else

template <typename TypeParam>
struct GPUDevice {
  using Dtype = TypeParam;
  static const Latte::Brew device = Latte::GPU;
};

template <typename Dtype>
class GPUDeviceTest : public MultiDeviceTest<GPUDevice<Dtype>> {};

using TestDtypesAndDevices =
    ::testing::Types<CPUDevice<float>, CPUDevice<double>, GPUDevice<float>,
                     GPUDevice<double>>;

#endif

}  // namespace latte

#endif  // LATTE_TEST_TEST_LATTE_MAIN_H_
