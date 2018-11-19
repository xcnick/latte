#ifndef LATTE_FILLER_H_
#define LATTE_FILLER_H_

#include "latte/blob.h"
#include "latte/proto/latte.pb.h"
#include "latte/util/math_functions.h"

namespace latte {

template <typename Dtype>
class Filler {
 public:
  explicit Filler(const FillerParameter &param) : filler_param_(param) {}
  virtual ~Filler() {}
  virtual void Fill(Blob<Dtype> *blob) = 0;

 protected:
  FillerParameter filler_param_;
};

template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
 public:
  explicit UniformFiller(const FillerParameter &param) : Filler<Dtype>(param) {}

  virtual void Fill(Blob<Dtype> *blob) {
    CHECK(blob->count());
    latte_rng_uniform<Dtype>(blob->count(), Dtype(this->filler_param_.min()),
                             Dtype(this->filler_param_.max()),
                             blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), 0)
        << "Sparsity not supported by this Filler.";
  }
};

}  // namespace latte

#endif