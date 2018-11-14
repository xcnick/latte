#ifndef LATTE_UTIL_MATH_FUNCTIONS_H_
#define LATTE_UTIL_MATH_FUNCTIONS_H_

#include <cstring>
#include "latte/common.h"

namespace latte {

inline void latte_memset(const size_t N, const int alpha, void *X) {
  std::memset(X, alpha, N);
}

void latte_gpu_memcpy(const size_t N, const void *X, void *Y);

void latte_gpu_memset(const size_t N, const int alpha, void *X);

}  // namespace latte

#endif