#ifndef LATTE_FILLER_H_
#define LATTE_FILLER_H_

#include "latte/blob.h"
#include "latte/proto/latte.pb.h"
#include "latte/util/math_functions.h"

namespace latte {

template <typename Dtype>
class Filler {
 public:
  explicit Filler(const FillerParameter &param) : filler_param_(param) {
    if (!filler_param_.has_type()) {
      filler_param_.mutable_type()->set_value("constant");
    }
    if (!filler_param_.has_value()) {
      filler_param_.mutable_value()->set_value(0.);
    }
    if (!filler_param_.has_min()) {
      filler_param_.mutable_min()->set_value(0.);
    }
    if (!filler_param_.has_max()) {
      filler_param_.mutable_max()->set_value(1.);
    }
    if (!filler_param_.has_mean()) {
      filler_param_.mutable_mean()->set_value(0.);
    }
    if (!filler_param_.has_std()) {
      filler_param_.mutable_std()->set_value(1.);
    }
    if (!filler_param_.has_sparse()) {
      filler_param_.mutable_sparse()->set_value(-1);
    }
  }
  virtual ~Filler() {}
  virtual void Fill(Blob<Dtype> *blob) = 0;

 protected:
  FillerParameter filler_param_;
};

template <typename Dtype>
class ConstantFiller : public Filler<Dtype> {
 public:
  explicit ConstantFiller(const FillerParameter &param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype> *blob) override {
    Dtype *data = blob->mutable_cpu_data();
    const int count = blob->count();
    const Dtype value = this->filler_param_.value().value();
    CHECK(count);
    for (int i = 0; i < count; ++i) {
      data[i] = value;
    }
    CHECK_EQ(this->filler_param_.sparse().value(), -1)
        << "Sparsity not supported by this Filler";
  }
};

template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
 public:
  explicit UniformFiller(const FillerParameter &param) : Filler<Dtype>(param) {}

  virtual void Fill(Blob<Dtype> *blob) override {
    CHECK(blob->count());
    latte_rng_uniform<Dtype>(
        blob->count(), Dtype(this->filler_param_.min().value()),
        Dtype(this->filler_param_.max().value()), blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse().value(), -1)
        << "Sparsity not supported by this Filler.";
  }
};

template <typename Dtype>
Filler<Dtype> *GetFiller(const FillerParameter &param) {
  if (!param.has_type()) {
    return new ConstantFiller<Dtype>(param);
  }
  const std::string &type = param.type().value();
  if (type == "constant") {
    return new ConstantFiller<Dtype>(param);
    // } else if (type == "gaussian") {
    //   return new GaussianFiller<Dtype>(param);
    // } else if (type == "positive_unitball") {
    //   return new PositiveUnitballFiller<Dtype>(param);
  } else if (type == "uniform") {
    return new UniformFiller<Dtype>(param);
    // } else if (type == "xavier") {
    //   return new XavierFiller<Dtype>(param);
    // } else if (type == "msra") {
    //   return new MSRAFiller<Dtype>(param);
    // } else if (type == "bilinear") {
    //   return new BilinearFiller<Dtype>(param);
  }
  return (Filler<Dtype> *)(NULL);
}

}  // namespace latte

#endif