#include "latte/util/math_functions.h"
#include <functional>
#include <random>
#include "latte/util/rng.h"

namespace latte {

template <>
void latte_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
                           const int N, const float alpha, const float *A,
                           const float *x, const float beta, float *y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void latte_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
                            const int N, const double alpha, const double *A,
                            const double *x, const double beta, double *y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void latte_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
                           const CBLAS_TRANSPOSE TransB, const int M,
                           const int N, const int K, const float alpha,
                           const float *A, const float *B, const float beta,
                           float *C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, N);
}

template <>
void latte_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
                            const CBLAS_TRANSPOSE TransB, const int M,
                            const int N, const int K, const double alpha,
                            const double *A, const double *B, const double beta,
                            double *C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, N);
}

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

template <>
void latte_cpu_axpby<float>(const int N, const float alpha, const float *X,
                            const float beta, float *Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void latte_cpu_axpby<double>(const int N, const double alpha, const double *X,
                             const double beta, double *Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
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
#ifdef USE_CUDA
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
void latte_set(const int N, const Dtype alpha, Dtype *Y) {
  if (alpha == 0) {
    std::memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(latte/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void latte_set<int>(const int N, const int alpha, int *Y);
template void latte_set<unsigned int>(const int N, const unsigned int alpha,
                                      unsigned int *Y);
template void latte_set<float>(const int N, const float alpha, float *Y);
template void latte_set<double>(const int N, const double alpha, double *Y);

template <>
void latte_add<float>(const int n, const float *a, const float *b, float *y) {
  vsAdd(n, a, b, y);
}

template <>
void latte_add<double>(const int n, const double *a, const double *b,
                       double *y) {
  vdAdd(n, a, b, y);
}

template <>
void latte_sub<float>(const int n, const float *a, const float *b, float *y) {
  vsAdd(n, a, b, y);
}

template <>
void latte_sub<double>(const int n, const double *a, const double *b,
                       double *y) {
  vdAdd(n, a, b, y);
}

template <>
void latte_mul<float>(const int n, const float *a, const float *b, float *y) {
  vsMul(n, a, b, y);
}

template <>
void latte_mul<double>(const int n, const double *a, const double *b,
                       double *y) {
  vdMul(n, a, b, y);
}

template <>
void latte_div<float>(const int n, const float *a, const float *b, float *y) {
  vsDiv(n, a, b, y);
}

template <>
void latte_div<double>(const int n, const double *a, const double *b,
                       double *y) {
  vdDiv(n, a, b, y);
}

template <>
void latte_exp<float>(const int n, const float *a, float *y) {
  vsExp(n, a, y);
}

template <>
void latte_exp<double>(const int n, const double *a, double *y) {
  vdExp(n, a, y);
}

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

template <typename Dtype>
void latte_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                        Dtype *r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  std::normal_distribution<Dtype> random_distribution(mu, sigma);
  std::function<Dtype()> variate_generator =
      bind(random_distribution, std::ref(*latte_rng()));
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template void latte_rng_gaussian<float>(const int n, const float mu,
                                        const float sigma, float *r);

template void latte_rng_gaussian<double>(const int n, const double mu,
                                         const double sigma, double *r);

template <typename Dtype>
void latte_rng_bernoulli(const int n, const Dtype p, int *r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  std::bernoulli_distribution random_distribution(p);
  std::function<Dtype()> variate_generator =
      bind(random_distribution, std::ref(*latte_rng()));
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template void latte_rng_bernoulli<double>(const int n, const double p, int *r);

template void latte_rng_bernoulli<float>(const int n, const float p, int *r);

template <typename Dtype>
void latte_rng_bernoulli(const int n, const Dtype p, unsigned int *r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  std::bernoulli_distribution random_distribution(p);
  std::function<Dtype()> variate_generator =
      bind(random_distribution, std::ref(*latte_rng()));
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template void latte_rng_bernoulli<double>(const int n, const double p,
                                          unsigned int *r);

template void latte_rng_bernoulli<float>(const int n, const float p,
                                         unsigned int *r);

}  // namespace latte
