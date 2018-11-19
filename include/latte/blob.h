#ifndef LATTE_BLOB_H_
#define LATTE_BLOB_H_

//#include <algorithm>
#include <string>
#include <vector>

#include "latte/common.h"
#include "latte/proto/latte.pb.h"
#include "latte/syncedmem.h"

const int kMaxBlobAxes = 32;

namespace latte {

template <typename Dtype>
class Blob : public Noncopyable {
 public:
  Blob() : data_(), diff_(), count_(0), capacity_(0) {}

  explicit Blob(const vector<int> &shape);

  void Reshape(const vector<int> &shape);
  void Reshape(const BlobShape &shape);
  void ReshapeLike(const Blob &other);

  inline string shape_string() const {
    ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }

  inline const vector<int> &shape() const { return shape_; }

  inline int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }

  inline int num_axes() const { return shape_.size(); }
  inline int count() const { return count_; }

  inline int count(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }

  inline int count(int start_axis) const {
    return count(start_axis, num_axes());
  }

  inline int CanonicalAxisIndex(int axis_index) const {
    CHECK_GE(axis_index, -num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << " -D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << " -D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }

  inline int LegacyShape(int index) const {
    CHECK_LE(num_axes(), 4)
        << "Cannot use legacy accessors on Blobs with > 4 axes";
    CHECK_LT(index, 4);
    CHECK_GE(index, -4);
    if (index >= num_axes() || index < -num_axes()) {
      return 1;
    }
    return shape(index);
  }

  inline int offset(const vector<int> &indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }

  void CopyFrom(const Blob<Dtype> &source, bool copy_diff = false,
                bool reshape = false);

  inline Dtype data_at(const vector<int> &index) const {
    return cpu_data()[offset(index)];
  }

  inline Dtype diff_at(const vector<int> &index) const {
    return cpu_diff()[offset(index)];
  }

  inline const shared_ptr<SyncedMemory> &data() const {
    CHECK(data_);
    return data_;
  }

  inline const shared_ptr<SyncedMemory> &diff() const {
    CHECK(diff_);
    return diff_;
  }

  const Dtype *cpu_data() const;
  void set_cpu_data(Dtype *type);
  const int *gpu_shape() const;
  const Dtype *gpu_data() const;
  void set_gpu_data(Dtype* data);
  const Dtype* cpu_diff() const;
  const Dtype* gpu_diff() const;
  Dtype* mutable_cpu_data();
  Dtype* mutable_gpu_data();
  Dtype* mutable_cpu_diff();
  Dtype* mutable_gpu_diff();
  void Update();
  void FromProto(const BlobProto &proto, bool reshape = true);
  void ToProto(BlobProto *proto, bool write_diff = false) const;

  // L1 norm
  Dtype asum_data() const;
  Dtype asum_diff() const;
  // L2 norm squared
  Dtype sumsq_data() const;
  Dtype sumsq_diff() const;

  void scale_data(Dtype scale_factor);
  void scale_diff(Dtype scale_factor);

  void ShareData(const Blob &other);
  void ShareDiff(const Blob &other);

  bool ShapeEquals(const BlobProto &other);

 protected:
  shared_ptr<SyncedMemory> data_;
  shared_ptr<SyncedMemory> diff_;
  shared_ptr<SyncedMemory> shape_data_;
  vector<int> shape_;
  int count_;
  int capacity_;
};

}  // namespace latte

#endif  // LATTE_BLOB_H_