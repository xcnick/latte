#include "latte/latte.h"
#include "latte/test/test_latte_main.h"

namespace latte {
#ifdef WITH_CUDA
  cudaDeviceProp LATTE_TEST_CUDA_PROP;
#endif
}

#ifdef WITH_CUDA
using latte::LATTE_TEST_CUDA_PROP;
#endif

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  latte::GlobalInit(&argc, &argv);
#ifdef WITH_CUDA
  // Before starting testing, let's first print out a few cuda device info.
  int device;
  cudaGetDeviceCount(&device);
  cout << "Cuda number of devices: " << device << endl;
  if (argc > 1) {
    // Use the given device
    device = atoi(argv[1]);
    cudaSetDevice(device);
    cout << "Setting to use device " << device << endl;
  } else if (CUDA_TEST_DEVICE >= 0) {
    // Use the device assigned in build configuration; but with a lower priority
    device = CUDA_TEST_DEVICE;
  }
  cudaGetDevice(&device);
  cout << "Current device id: " << device << endl;
  cudaGetDeviceProperties(&LATTE_TEST_CUDA_PROP, device);
  cout << "Current device name: " << LATTE_TEST_CUDA_PROP.name << endl;
#endif
  // invoke the test.
  return RUN_ALL_TESTS();
}
