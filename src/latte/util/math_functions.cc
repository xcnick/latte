#include "latte/util/math_functions.h"
#include <random>
#include "latte/util/rng.h"

namespace latte {

template <>
void latte_axpy<float>(const int N, const float alpha, const float *X,
                       float *Y) {
  cblas_saxpy(N, alpha, X, 1, Y, 1);
}

template <>
void latte_axpy<double>(const int N, const double alpha, const double *X,
                        double *Y) {
  cblas_daxpy(N, alpha, X, 1, Y, 1);
}

template <typename Dtype>
Dtype latte_cpu_dot(const int n, const Dtype *x, const Dtype *y) {
  return latte_cpu_strided_dot(n, x, 1, y, 1);
}

template float latte_cpu_dot<float>(const int n, const float *x,
                                    const float *y);

template double latte_cpu_dot<double>(const int n, const double *x,
                                      const double *y);

template <>
float latte_cpu_strided_dot<float>(const int n, const float *x, const int incx,
                                   const float *y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double latte_cpu_strided_dot<double>(const int n, const double *x,
                                     const int incx, const double *y,
                                     const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <>
void latte_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void latte_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void latte_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float *y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void latte_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double *y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

template <>
float latte_cpu_asum<float>(const int n, const float *x) {
  return cblas_sasum(n, x, 1);
}

template <>
double latte_cpu_asum<double>(const int n, const double *x) {
  return cblas_dasum(n, x, 1);
}

template <typename Dtype>
void latte_copy(const int N, const Dtype *X, Dtype *Y) {
  if (X != Y) {
    if (Latte::mode() == Latte::GPU) {
#ifndef CPU_ONLY
      latte_gpu_memcpy(sizeof(Dtype) * N, X, Y);
#else
      NO_GPU;
#endif
    } else {
      std::memcpy(Y, X, sizeof(Dtype) * N);
    }
  }
}

template void latte_copy<int>(const int N, const int *X, int *Y);
template void latte_copy<unsigned int>(const int N, const unsigned int *X,
                                       unsigned int *Y);
template void latte_copy<float>(const int N, const float *X, float *Y);
template void latte_copy<double>(const int N, const double *X, double *Y);

template <typename Dtype>
Dtype latte_nextafter(const Dtype b) {
  return std::nextafter(b, std::numeric_limits<Dtype>::max());
}

template float latte_nextafter(const float b);

template double latte_nextafter(const double b);

template <typename Dtype>
void latte_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype *r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  std::uniform_real_distribution<Dtype> random_distribution(
      a, latte_nextafter<Dtype>(b));
  std::function<Dtype()> variate_generator =
      bind(random_distribution, std::ref(*latte_rng()));
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template void latte_rng_uniform<float>(const int n, const float a,
                                       const float b, float *r);

template void latte_rng_uniform<double>(const int n, const double a,
                                        const double b, double *r);

}  // namespace latte
