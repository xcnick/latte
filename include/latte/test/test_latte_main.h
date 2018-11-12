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

#ifdef CMAKE_BUILD
  #include "latte_config.h"
#else
  #define CUDA_TEST_DEVICE -1
  #define EXAMPLES_SOURCE_DIR "examples/"
  #define ABS_TEST_DATA_DIR "src/latte/test/test_data"
#endif

int main(int argc, char** argv);

// namespace latte {

// template <typename TypeParam>
// class MultiDeviceTest : public ::testing::Test {
//  public:
//   typedef typename TypeParam::Dtype Dtype;
//  protected:
//   MultiDeviceTest() {
//     Latte::set_mode(TypeParam::device);
//   }
//   virtual ~MultiDeviceTest() {}
// };

// typedef ::testing::Types<float, double> TestDtypes;

// template <typename TypeParam>
// struct CPUDevice {
//   typedef TypeParam Dtype;
//   static const Latte::Brew device = Latte::CPU;
// };

// template <typename Dtype>
// class CPUDeviceTest : public MultiDeviceTest<CPUDevice<Dtype> > {
// };

// #ifdef CPU_ONLY

// typedef ::testing::Types<CPUDevice<float>,
//                          CPUDevice<double> > TestDtypesAndDevices;

// #else

// template <typename TypeParam>
// struct GPUDevice {
//   typedef TypeParam Dtype;
//   static const Latte::Brew device = Latte::GPU;
// };

// template <typename Dtype>
// class GPUDeviceTest : public MultiDeviceTest<GPUDevice<Dtype> > {
// };

// typedef ::testing::Types<CPUDevice<float>, CPUDevice<double>,
//                          GPUDevice<float>, GPUDevice<double> >
//                          TestDtypesAndDevices;

// #endif

// }  // namespace latte

#endif  // LATTE_TEST_TEST_LATTE_MAIN_H_
