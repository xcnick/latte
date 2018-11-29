#ifndef LATTE_UTIL_MATH_FUNCTIONS_H_
#define LATTE_UTIL_MATH_FUNCTIONS_H_

#include <cstring>
extern "C" {
#include <cblas.h>
}
#include "latte/common.h"

namespace latte {

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
void latte_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

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
void latte_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

#ifndef CPU_ONLY

void latte_gpu_memcpy(const size_t N, const void *X, void *Y);

void latte_gpu_memset(const size_t N, const int alpha, void *X);

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