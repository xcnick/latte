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
    if (filler_param_.type_oneof_case() == FillerParameter::TypeOneofCase::TYPE_ONEOF_NOT_SET) {
      filler_param_.set_type("constant");
    }
    if (filler_param_.max_oneof_case() == FillerParameter::MaxOneofCase::MAX_ONEOF_NOT_SET) {
      filler_param_.set_max(1.);
    }
    if (filler_param_.std_oneof_case() == FillerParameter::StdOneofCase::STD_ONEOF_NOT_SET) {
      filler_param_.set_std(1.);
    }
    if (filler_param_.sparse_oneof_case() == FillerParameter::SparseOneofCase::SPARSE_ONEOF_NOT_SET) {
      filler_param_.set_sparse(-1);
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
    const Dtype value = this->filler_param_.value();
    CHECK(count);
    for (int i = 0; i < count; ++i) {
      data[i] = value;
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
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
        blob->count(), Dtype(this->filler_param_.min()),
        Dtype(this->filler_param_.max()), blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
        << "Sparsity not supported by this Filler.";
  }
};

template <typename Dtype>
class GaussianFiller : public Filler<Dtype> {
 public:
  explicit GaussianFiller(const FillerParameter &param)
      : Filler<Dtype>(param) {}

  virtual void Fill(Blob<Dtype> *blob) {
    Dtype *data = blob->mutable_cpu_data();
    CHECK(blob->count());
    latte_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_.mean()),
                              Dtype(this->filler_param_.std()),
                              blob->mutable_cpu_data());
    int sparse = this->filler_param_.sparse();
    CHECK_GE(sparse, -1);
    if (sparse >= 0) {
      // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
      // These have num == channels == 1; width is number of inputs; height is
      // number of outputs.  The 'sparse' variable specifies the mean number
      // of non-zero input weights for a given output.
      CHECK_GE(blob->num_axes(), 1);
      const int num_outputs = blob->shape(0);
      Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
      rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(int)));
      int *mask = reinterpret_cast<int *>(rand_vec_->mutable_cpu_data());
      latte_rng_bernoulli(blob->count(), non_zero_probability, mask);
      for (int i = 0; i < blob->count(); ++i) {
        data[i] *= mask[i];
      }
    }
  }

 protected:
  shared_ptr<SyncedMemory> rand_vec_;
};

template <typename Dtype>
Filler<Dtype> *GetFiller(const FillerParameter &param) {
  if (param.type().empty()) {
    return new ConstantFiller<Dtype>(param);
  }
  const std::string &type = param.type();
  if (type == "constant") {
    return new ConstantFiller<Dtype>(param);
  } else if (type == "gaussian") {
    return new GaussianFiller<Dtype>(param);
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