#include <cuda_runtime.h>
#include "latte/util/math_functions.h"

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
void latte_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
                           const CBLAS_TRANSPOSE TransB, const int M,
                           const int N,  int K, const float alpha,
                           const float *A, const float *B, const float beta,
                           float *C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Latte::cublas_handle(), cuTransB, cuTransA, N, M, K,
                           &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void latte_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
                            const CBLAS_TRANSPOSE TransB, const int M,
                            const int N, const int K, const double alpha,
                            const double *A, const double *B, const double beta,
                            double *C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Latte::cublas_handle(), cuTransB, cuTransA, N, M, K,
                           &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void latte_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
                           const int N, const float alpha, const float *A,
                           const float *x, const float beta, float *y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Latte::cublas_handle(), cuTransA, N, M, &alpha, A, N,
                           x, 1, &beta, y, 1));
}

template <>
void latte_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
                            const int N, const double alpha, const double *A,
                            const double *x, const double beta, double *y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Latte::cublas_handle(), cuTransA, N, M, &alpha, A, N,
                           x, 1, &beta, y, 1));
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

}  // namespace latte