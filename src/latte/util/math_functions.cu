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

void latte_gpu_memset(const size_t N, const int alpha, void *X) {
  CUDA_CHECK(cudaMemset(X, alpha, N));  // NOLINT
}

template <>
void latte_gpu_axpy<float>(const int N, const float alpha, const float *X,
                           float *Y) {
  CUBLAS_CHECK(cublasSaxpy(Latte::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void latte_gpu_axpy<double>(const int N, const double alpha, const double *X,
                            double *Y) {
  CUBLAS_CHECK(cublasDaxpy(Latte::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void latte_gpu_dot<float>(const int n, const float *x, const float *y,
                          float *out) {
  CUBLAS_CHECK(cublasSdot(Latte::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void latte_gpu_dot<double>(const int n, const double *x, const double *y,
                           double *out) {
  CUBLAS_CHECK(cublasDdot(Latte::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void latte_gpu_asum<float>(const int n, const float *x, float *y) {
  CUBLAS_CHECK(cublasSasum(Latte::cublas_handle(), n, x, 1, y));
}

template <>
void latte_gpu_asum<double>(const int n, const double *x, double *y) {
  CUBLAS_CHECK(cublasDasum(Latte::cublas_handle(), n, x, 1, y));
}

template <>
void latte_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Latte::cublas_handle(), N, &alpha, X, 1));
}

template <>
void latte_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Latte::cublas_handle(), N, &alpha, X, 1));
}

}