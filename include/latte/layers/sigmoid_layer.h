#ifndef LATTE_SIGMOID_LAYER_H_
#define LATTE_SIGMOID_LAYER_H_

#include "latte/blob.h"
#include "latte/layer.h"
#include "latte/proto/latte.pb.h"

#include "latte/layers/neuron_layer.h"

namespace latte {
template <typename Dtype>
class SigmoidLayer : public NeuronLayer<Dtype> {
 public:
  explicit SigmoidLayer(const LayerParameter &param)
      : NeuronLayer<Dtype>(param) {}

  virtual inline const char *type() const override { return "Sigmoid"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top) override;
  virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top) override;
};
}  // namespace latte

#endif  // LATTE_SIGMOID_LAYER_H_