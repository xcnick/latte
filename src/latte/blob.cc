#include "latte/blob.h"
#include "latte/common.h"
#include "latte/syncedmem.h"
#include "latte/util/math_functions.h"

namespace latte {

template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int> &shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes);
  count_ = 1;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
  }
  int *shape_data = reinterpret_cast<int *>(shape_data_->mutable_cpu_data());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    if (count_ != 0) {
      CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    }
    count_ *= shape[i];
    shape_[i] = shape[i];
    shape_data[i] = shape[i];
  }
  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape &shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype> &other) {
  Reshape(other.shape());
}

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int> &shape)
    // capacity_ must be initialized before calling Reshape
    : capacity_(0) {
  Reshape(shape);
}

template <typename Dtype>
const int *Blob<Dtype>::gpu_shape() const {
  CHECK(shape_data_);
  return reinterpret_cast<const int *>(shape_data_->gpu_data());
}

template <typename Dtype>
const Dtype *Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return reinterpret_cast<const Dtype *>(data_->cpu_data());
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype *data) {
  CHECK(data);
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  }
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype *Blob<Dtype>::gpu_data() const {
  CHECK(data_);
  return reinterpret_cast<const Dtype *>(data_->gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::set_gpu_data(Dtype *data) {
  CHECK(data);
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  }
  data_->set_gpu_data(data);
}

template <typename Dtype>
const Dtype *Blob<Dtype>::cpu_diff() const {
  CHECK(diff_);
  return reinterpret_cast<const Dtype *>(diff_->cpu_data());
}

template <typename Dtype>
const Dtype *Blob<Dtype>::gpu_diff() const {
  CHECK(diff_);
  return reinterpret_cast<const Dtype *>(diff_->gpu_data());
}

template <typename Dtype>
Dtype *Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return reinterpret_cast<Dtype *>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype *Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return reinterpret_cast<Dtype *>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype *Blob<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return reinterpret_cast<Dtype *>(diff_->mutable_cpu_data());
}

template <typename Dtype>
Dtype *Blob<Dtype>::mutable_gpu_diff() {
  CHECK(diff_);
  return reinterpret_cast<Dtype *>(diff_->mutable_gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob &other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob &other) {
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

template <>
void Blob<unsigned int>::Update() {
  NOT_IMPLEMENTED;
}

template <>
void Blob<int>::Update() {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::Update() {
  switch (data_->head()) {
    case SyncedMemory::SyncedHead::HEAD_AT_CPU:
      latte_axpy<Dtype>(count_, Dtype(-1),
                        reinterpret_cast<const Dtype *>(diff_->cpu_data()),
                        reinterpret_cast<Dtype *>(data_->mutable_cpu_data()));
      break;
    case SyncedMemory::SyncedHead::HEAD_AT_GPU:
    case SyncedMemory::SyncedHead::SYNCED:
#ifndef CPU_ONLY
      latte_gpu_axpy<Dtype>(
          count_, Dtype(-1), reinterpret_cast<const Dtype *>(diff_->gpu_data()),
          reinterpret_cast<Dtype *>(data_->mutable_gpu_data()));
#else
      NO_GPU;
#endif
      break;
    default:
      LOG(FATAL) << "Syncedmem not initialized.";
  }
}

template <>
unsigned int Blob<unsigned int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <>
int Blob<int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_data() const {
  if (!data_) {
    return 0;
  }
  switch (data_->head()) {
    case SyncedMemory::SyncedHead::HEAD_AT_CPU:
      return latte_cpu_asum(count_, cpu_data());
    case SyncedMemory::SyncedHead::HEAD_AT_GPU:
    case SyncedMemory::SyncedHead::SYNCED:
#ifndef CPU_ONLY
      Dtype asum;
      latte_gpu_asum(count_, gpu_data(), &asum);
      return asum;
#else
      NO_GPU;
#endif
    case SyncedMemory::SyncedHead::UNINITIALIZED:
      return 0;
    default:
      LOG(FATAL) << "Unknown SyncedMemory head state.";
  }
  return 0;
}

template <>
unsigned int Blob<unsigned int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <>
int Blob<int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_diff() const {
  if (!diff_) {
    return 0;
  }
  switch (diff_->head()) {
    case SyncedMemory::SyncedHead::HEAD_AT_CPU:
      return latte_cpu_asum(count_, cpu_diff());
    case SyncedMemory::SyncedHead::HEAD_AT_GPU:
    case SyncedMemory::SyncedHead::SYNCED:
#ifndef CPU_ONLY
      Dtype asum;
      latte_gpu_asum(count_, gpu_diff(), &asum);
      return asum;
#else
      NO_GPU;
#endif
    case SyncedMemory::SyncedHead::UNINITIALIZED:
      return 0;
    default:
      LOG(FATAL) << "Unknown SyncedMemory head state.";
  }
  return 0;
}

template <>
unsigned int Blob<unsigned int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <>
int Blob<int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_data() const {
  Dtype sumsq;
  const Dtype *data;
  if (!data_) {
    return 0;
  }
  switch (data_->head()) {
    case SyncedMemory::SyncedHead::HEAD_AT_CPU:
      data = cpu_data();
      sumsq = latte_cpu_dot(count_, data, data);
      break;
    case SyncedMemory::SyncedHead::HEAD_AT_GPU:
    case SyncedMemory::SyncedHead::SYNCED:
#ifndef CPU_ONLY
      data = gpu_data();
      latte_gpu_dot(count_, data, data, &sumsq);
#else
      NO_GPU;
#endif
      break;
    case SyncedMemory::SyncedHead::UNINITIALIZED:
      return 0;
    default:
      LOG(FATAL) << "Unknown SyncedMemory head state.";
  }
  return sumsq;
}

template <>
unsigned int Blob<unsigned int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <>
int Blob<int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
  Dtype sumsq;
  const Dtype *diff;
  if (!diff_) {
    return 0;
  }
  switch (diff_->head()) {
    case SyncedMemory::SyncedHead::HEAD_AT_CPU:
      diff = cpu_diff();
      sumsq = latte_cpu_dot(count_, diff, diff);
      break;
    case SyncedMemory::SyncedHead::HEAD_AT_GPU:
    case SyncedMemory::SyncedHead::SYNCED:
#ifndef CPU_ONLY
      diff = gpu_diff();
      latte_gpu_dot(count_, diff, diff, &sumsq);
#else
      NO_GPU;
#endif
      break;
    case SyncedMemory::SyncedHead::UNINITIALIZED:
      return 0;
    default:
      LOG(FATAL) << "Unknown SyncedMemory head state.";
  }
  return sumsq;
}

template <>
void Blob<unsigned int>::scale_data(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <>
void Blob<int>::scale_data(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor) {
  Dtype *data;
  if (!data_) {
    return;
  }
  switch (data_->head()) {
    case SyncedMemory::SyncedHead::HEAD_AT_CPU:
      data = mutable_cpu_data();
      latte_scal(count_, scale_factor, data);
      return;
    case SyncedMemory::SyncedHead::HEAD_AT_GPU:
    case SyncedMemory::SyncedHead::SYNCED:
#ifndef CPU_ONLY
      data = mutable_gpu_data();
      latte_gpu_scal(count_, scale_factor, data);
#else
      NO_GPU;
#endif
      return;
    case SyncedMemory::SyncedHead::UNINITIALIZED:
      return;
    default:
      LOG(FATAL) << "Unknown SyncedMemory head state.";
  }
}

template <>
void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <>
void Blob<int>::scale_diff(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
  Dtype *diff;
  if (!diff_) {
    return;
  }
  switch (diff_->head()) {
    case SyncedMemory::SyncedHead::HEAD_AT_CPU:
      diff = mutable_cpu_diff();
      latte_scal(count_, scale_factor, diff);
      return;
    case SyncedMemory::SyncedHead::HEAD_AT_GPU:
    case SyncedMemory::SyncedHead::SYNCED:
#ifndef CPU_ONLY
      diff = mutable_gpu_diff();
      latte_gpu_scal(count_, scale_factor, diff);
#else
      NO_GPU;
#endif
      return;
    case SyncedMemory::SyncedHead::UNINITIALIZED:
      return;
    default:
      LOG(FATAL) << "Unknown SyncedMemory head state.";
  }
}

template <typename Dtype>
bool Blob<Dtype>::ShapeEquals(const BlobProto &other) {
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob &source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Tring to copy blobs of different sizes.";
    }
  }
  switch (Latte::mode()) {
    case Latte::GPU:
      if (copy_diff) {
        latte_copy(count_, source.gpu_diff(),
                   reinterpret_cast<Dtype *>(diff_->mutable_gpu_data()));
      } else {
        latte_copy(count_, source.gpu_data(),
                   reinterpret_cast<Dtype *>(data_->mutable_gpu_data()));
      }
      break;
    case Latte::CPU:
      if (copy_diff) {
        latte_copy(count_, source.cpu_diff(),
                   reinterpret_cast<Dtype *>(diff_->mutable_cpu_data()));
      } else {
        latte_copy(count_, source.cpu_data(),
                   reinterpret_cast<Dtype *>(data_->mutable_cpu_data()));
      }
      break;
    default:
      LOG(FATAL) << "Unknown latte mode.";
  }
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto &proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
    shape.resize(proto.shape().dim_size());
    for (int i = 0; i < proto.shape().dim_size(); ++i) {
      shape[i] = proto.shape().dim(i);
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }

  Dtype *data_vec = mutable_cpu_data();
  if (proto.double_data_size() > 0) {
    CHECK_EQ(count_, proto.double_data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.double_data(i);
    }
  } else {
    CHECK_EQ(count_, proto.data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.data(i);
    }
  }
  Dtype *diff_vec = mutable_cpu_diff();
  if (proto.double_diff_size() > 0) {
    CHECK_EQ(count_, proto.double_diff_size());
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.double_diff(i);
    }
  } else {
    CHECK_EQ(count_, proto.diff_size());
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
}

template <>
void Blob<float>::ToProto(BlobProto *proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  const float *data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
  if (write_diff) {
    const float *diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

template <>
void Blob<double>::ToProto(BlobProto *proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  const double *data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_double_data(data_vec[i]);
  }
  if (write_diff) {
    const double *diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
  }
}

INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;
}  // namespace latte