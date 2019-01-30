#ifndef LATTE_EUCLIDEAN_LOSS_LAYER_H_
#define LATTE_EUCLIDEAN_LOSS_LAYER_H_

#include "latte/blob.h"
#include "latte/layer.h"
#include "latte/proto/latte.pb.h"

#include "latte/layers/loss_layer.h"

namespace latte {

template <typename Dtype>
class EuclideanLossLayer : public LossLayer<Dtype> {
 public:
  explicit EuclideanLossLayer(const LayerParameter &param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "EuclideanLoss"; }

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);

  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom);

  virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom);
  Blob<Dtype> diff_;
};

}  // namespace latte

#endif  //  LATTE_EUCLIDEAN_LOSS_LAYER_H_