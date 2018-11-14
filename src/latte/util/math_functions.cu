#include "latte/util/math_functions.h"
#include <cuda_runtime.h>

#include "latte/common.h"
#include "latte/util/device_alternate.h"

namespace latte {

void latte_gpu_memcpy(const size_t N, const void *X, void *Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));
  }
}

void latte_gpu_memset(const size_t N, const int alpha, void* X) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaMemset(X, alpha, N));  // NOLINT
#else
  NO_GPU;
#endif
}

}