#ifndef LATTE_UTIL_DEVICE_ALTERNATE_H_
#define LATTE_UTIL_DEVICE_ALTERNATE_H_

#ifndef WITH_CUDA  // CPU-only Latte.

#include <vector>

// Stub out GPU calls as unavailable.

#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Latte: check mode."

#define STUB_GPU(classname)                                                  \
  template <typename Dtype>                                                  \
  void classname<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,    \
                                     const vector<Blob<Dtype> *> &top) {     \
    NO_GPU;                                                                  \
  }                                                                          \
  template <typename Dtype>                                                  \
  void classname<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,      \
                                      const vector<bool> &propagate_down,    \
                                      const vector<Blob<Dtype> *> &bottom) { \
    NO_GPU;                                                                  \
  }

#define STUB_GPU_FORWARD(classname, funcname)                                  \
  template <typename Dtype>                                                    \
  void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype> *> &bottom, \
                                          const vector<Blob<Dtype> *> &top) {  \
    NO_GPU;                                                                    \
  }

#define STUB_GPU_BACKWARD(classname, funcname)                              \
  template <typename Dtype>                                                 \
  void classname<Dtype>::funcname##_##gpu(                                  \
      const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down, \
      const vector<Blob<Dtype> *> &bottom) {                                \
    NO_GPU;                                                                 \
  }

#else  // Normal GPU + CPU Latte.

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#ifdef USE_CUDNN           // cuDNN acceleration library.
#include "latte/util/cudnn.h"
#endif

//
// CUDA macros
//

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition)                                         \
  /* Code block avoids redefinition of cudaError_t error */           \
  do {                                                                \
    cudaError_t error = condition;                                    \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition)                        \
  do {                                                 \
    cublasStatus_t status = condition;                 \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS)            \
        << " " << latte::cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition)                        \
  do {                                                 \
    curandStatus_t status = condition;                 \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS)            \
        << " " << latte::curandGetErrorString(status); \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

namespace latte {

// CUDA: library error reporting.
const char *cublasGetErrorString(cublasStatus_t error);
const char *curandGetErrorString(curandStatus_t error);

// CUDA: use 512 threads per block
const int LATTE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int LATTE_GET_BLOCKS(const int N) {
  return (N + LATTE_CUDA_NUM_THREADS - 1) / LATTE_CUDA_NUM_THREADS;
}

}  // namespace latte

#endif  // WITH_CUDA

#endif  // Latte_UTIL_DEVICE_ALTERNATE_H_
