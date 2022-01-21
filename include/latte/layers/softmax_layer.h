#ifndef LATTE_SOFTMAX_LAYER_H_
#define LATTE_SOFTMAX_LAYER_H_

#include "latte/blob.h"
#include "latte/layer.h"
#include "latte/proto/latte.pb.h"

namespace latte {

template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxLayer(const LayerParameter &param) : Layer<Dtype>(param) {
    if (this->layer_param_.softmax_param().axis_oneof_case() ==
        SoftmaxParameter::AxisOneofCase::AXIS_ONEOF_NOT_SET) {
      this->layer_param_.mutable_softmax_param()->set_axis(1);
    }
  }

  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "Softmax"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);

  int outer_num_;     // 外层个数，即batch_size
  int inner_num_;     // 内层个数，默认1
  int softmax_axis_;  // 默认1

  Blob<Dtype> sum_multiplier_;    // 求和缓存
  Blob<Dtype> scale_;             // 输出结果缓存
};
}  // namespace latte

#endif  // LATTE_SOFTMAX_LAYER_H_