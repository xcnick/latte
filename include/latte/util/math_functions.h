#ifndef LATTE_UTIL_MATH_FUNCTIONS_H_
#define LATTE_UTIL_MATH_FUNCTIONS_H_

#include <cstring>
extern "C" {
#include <cblas.h>
}
#include "latte/common.h"

namespace latte {

// Functions that caffe uses but are not present if MKL is not linked.

// A simple way to define the vsl unary functions. The operation should
// be in the form e.g. y[i] = sqrt(a[i])
#define DEFINE_VSL_UNARY_FUNC(name, operation)                    \
  template <typename Dtype>                                       \
  void v##name(const int n, const Dtype *a, Dtype *y) {           \
    CHECK_GT(n, 0);                                               \
    CHECK(a);                                                     \
    CHECK(y);                                                     \
    for (int i = 0; i < n; ++i) {                                 \
      operation;                                                  \
    }                                                             \
  }                                                               \
  inline void vs##name(const int n, const float *a, float *y) {   \
    v##name<float>(n, a, y);                                      \
  }                                                               \
  inline void vd##name(const int n, const double *a, double *y) { \
    v##name<double>(n, a, y);                                     \
  }

DEFINE_VSL_UNARY_FUNC(Sqr, y[i] = a[i] * a[i])
DEFINE_VSL_UNARY_FUNC(Sqrt, y[i] = sqrt(a[i]))
DEFINE_VSL_UNARY_FUNC(Exp, y[i] = exp(a[i]))
DEFINE_VSL_UNARY_FUNC(Ln, y[i] = log(a[i]))
DEFINE_VSL_UNARY_FUNC(Abs, y[i] = fabs(a[i]))

// A simple way to define the vsl unary functions with singular parameter b.
// The operation should be in the form e.g. y[i] = pow(a[i], b)
#define DEFINE_VSL_UNARY_FUNC_WITH_PARAM(name, operation)                      \
  template <typename Dtype>                                                    \
  void v##name(const int n, const Dtype *a, const Dtype b, Dtype *y) {         \
    CHECK_GT(n, 0);                                                            \
    CHECK(a);                                                                  \
    CHECK(y);                                                                  \
    for (int i = 0; i < n; ++i) {                                              \
      operation;                                                               \
    }                                                                          \
  }                                                                            \
  inline void vs##name(const int n, const float *a, const float b, float *y) { \
    v##name<float>(n, a, b, y);                                                \
  }                                                                            \
  inline void vd##name(const int n, const double *a, const float b,            \
                       double *y) {                                            \
    v##name<double>(n, a, b, y);                                               \
  }

DEFINE_VSL_UNARY_FUNC_WITH_PARAM(Powx, y[i] = pow(a[i], b))

// A simple way to define the vsl binary functions. The operation should
// be in the form e.g. y[i] = a[i] + b[i]
#define DEFINE_VSL_BINARY_FUNC(name, operation)                         \
  template <typename Dtype>                                             \
  void v##name(const int n, const Dtype *a, const Dtype *b, Dtype *y) { \
    CHECK_GT(n, 0);                                                     \
    CHECK(a);                                                           \
    CHECK(b);                                                           \
    CHECK(y);                                                           \
    for (int i = 0; i < n; ++i) {                                       \
      operation;                                                        \
    }                                                                   \
  }                                                                     \
  inline void vs##name(const int n, const float *a, const float *b,     \
                       float *y) {                                      \
    v##name<float>(n, a, b, y);                                         \
  }                                                                     \
  inline void vd##name(const int n, const double *a, const double *b,   \
                       double *y) {                                     \
    v##name<double>(n, a, b, y);                                        \
  }

DEFINE_VSL_BINARY_FUNC(Add, y[i] = a[i] + b[i])
DEFINE_VSL_BINARY_FUNC(Sub, y[i] = a[i] - b[i])
DEFINE_VSL_BINARY_FUNC(Mul, y[i] = a[i] * b[i])
DEFINE_VSL_BINARY_FUNC(Div, y[i] = a[i] / b[i])

inline void cblas_saxpby(const int N, const float alpha, const float *X,
                         const int incX, const float beta, float *Y,
                         const int incY) {
  cblas_sscal(N, beta, Y, incY);
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}
inline void cblas_daxpby(const int N, const double alpha, const double *X,
                         const int incX, const double beta, double *Y,
                         const int incY) {
  cblas_dscal(N, beta, Y, incY);
  cblas_daxpy(N, alpha, X, incX, Y, incY);
}

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

template <typename Dtype>
void latte_cpu_axpby(const int N, const Dtype alpha, const Dtype *X,
                     const Dtype beta, Dtype *Y);

template <typename Dtype>
void latte_copy(const int N, const Dtype *X, Dtype *Y);

template <typename Dtype>
void latte_set(const int N, const Dtype alpha, Dtype *X);

inline void latte_memset(const size_t N, const int alpha, void *X) {
  std::memset(X, alpha, N);
}

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
void latte_add(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void latte_sub(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void latte_mul(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void latte_div(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void latte_exp(const int N, const Dtype* a, Dtype* y);

template <typename Dtype>
Dtype latte_nextafter(const Dtype b);

template <typename Dtype>
void latte_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype *r);

template <typename Dtype>
void latte_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                        Dtype *r);

template <typename Dtype>
void latte_rng_bernoulli(const int n, const Dtype p, int *r);

template <typename Dtype>
void latte_rng_bernoulli(const int n, const Dtype p, unsigned int *r);

#ifdef WITH_CUDA

void latte_gpu_memcpy(const size_t N, const void *X, void *Y);

template <typename Dtype>
void latte_gpu_set(const int N, const Dtype alpha, Dtype *X);

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
void latte_gpu_axpby(const int N, const Dtype alpha, const Dtype *X,
                     const Dtype beta, Dtype *Y);

template <typename Dtype>
void latte_gpu_dot(const int n, const Dtype *x, const Dtype *y, Dtype *out);

template <typename Dtype>
void latte_gpu_asum(const int n, const Dtype *x, Dtype *y);

template <typename Dtype>
void latte_gpu_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void latte_gpu_add(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void latte_gpu_sub(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void latte_gpu_mul(const int N, const Dtype *a, const Dtype *b, Dtype *y);

template <typename Dtype>
void latte_gpu_div(const int N, const Dtype *a, const Dtype *b, Dtype *y);

#endif
}  // namespace latte

#endif