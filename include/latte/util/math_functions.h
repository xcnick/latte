#ifndef LATTE_UTIL_MATH_FUNCTIONS_H_
#define LATTE_UTIL_MATH_FUNCTIONS_H_

#include <cstring>
extern "C" {
#include <cblas.h>
}
#include "latte/common.h"

namespace latte {

// C=alpha*A*B+beta*C
// 当alpha = 1, beta = 0 时，相当于C = A * B
template <typename Dtype>
void latte_cpu_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                    const int M, const int N, const int K, const Dtype alpha,
                    const Dtype *A, const Dtype *B, const Dtype beta, Dtype *C);

// y=alpha*A*x+beta*y
// x和y是向量，A是矩阵(M * N)
template <typename Dtype>
void latte_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                    const Dtype alpha, const Dtype *A, const Dtype *x,
                    const Dtype beta, Dtype *y);

template <typename Dtype>
void latte_axpy(const int N, const Dtype alpha, const Dtype *X, Dtype *Y);

// Returns the sum of the absolute values of the elements of vector x
template <typename Dtype>
Dtype latte_cpu_asum(const int n, const Dtype *x);

template <typename Dtype>
Dtype latte_cpu_dot(const int n, const Dtype *x, const Dtype *y);

template <typename Dtype>
Dtype latte_cpu_strided_dot(const int n, const Dtype *x, const int incx,
                            const Dtype *y, const int incy);

template <typename Dtype>
void latte_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void latte_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype *y);

template <typename Dtype>
void latte_copy(const int N, const Dtype *X, Dtype *Y);

template <typename Dtype>
void latte_set(const int N, const Dtype alpha, Dtype *X);

inline void latte_memset(const size_t N, const int alpha, void *X) {
  std::memset(X, alpha, N);
}

template <typename Dtype>
Dtype latte_nextafter(const Dtype b);

template <typename Dtype>
void latte_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype *r);

#ifndef CPU_ONLY

void latte_gpu_memcpy(const size_t N, const void *X, void *Y);

void latte_gpu_memset(const size_t N, const int alpha, void *X);

template <typename Dtype>
void latte_gpu_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                    const int M, const int N, const int K, const Dtype alpha,
                    const Dtype *A, const Dtype *B, const Dtype beta, Dtype *C);

template <typename Dtype>
void latte_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                    const Dtype alpha, const Dtype *A, const Dtype *x,
                    const Dtype beta, Dtype *y);

template <typename Dtype>
void latte_gpu_axpy(const int N, const Dtype alpha, const Dtype *X, Dtype *Y);

template <typename Dtype>
void latte_gpu_dot(const int n, const Dtype *x, const Dtype *y, Dtype *out);

template <typename Dtype>
void latte_gpu_asum(const int n, const Dtype *x, Dtype *y);

template <typename Dtype>
void latte_gpu_scal(const int N, const Dtype alpha, Dtype *X);

#endif
}  // namespace latte

#endif