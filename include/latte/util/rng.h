#ifndef LATTE_RNG_CPP_H_
#define LATTE_RNG_CPP_H_

#include <algorithm>
#include <iterator>
#include <random>

#include "latte/common.h"

namespace latte {

using rng_t = std::mt19937_64;

inline rng_t *latte_rng() {
  return static_cast<latte::rng_t *>(Latte::rng_stream().generator());
}

// Fisher–Yates algorithm
template <class RandomAccessIterator, class RandomGenerator>
inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end,
                    RandomGenerator *gen) {
  using difference_type =
      typename std::iterator_traits<RandomAccessIterator>::difference_type;
  using dist_type = typename std::uniform_int_distribution<difference_type>;

  difference_type length = std::distance(begin, end);
  if (length <= 0) return;

  for (difference_type i = length - 1; i > 0; --i) {
    dist_type dist(0, i);
    std::iter_swap(begin + i, begin + dist(*gen));
  }
}

template <class RandomAccessIterator>
inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end) {
  shuffle(begin, end, latte_rng());
}
}  // namespace latte

#endif  // LATTE_RNG_CPP_H_
