#ifndef LATTE_NEURON_LAYER_H_
#define LATTE_NEURON_LAYER_H_

#include "latte/blob.h"
#include "latte/layer.h"
#include "latte/proto/latte.pb.h"

namespace latte {

template <typename Dtype>
class NeuronLayer : public Layer<Dtype> {
 public:
  explicit NeuronLayer(const LayerParameter &param) : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
};

}  // namespace latte

#endif  // LATTE_NEURON_LAYER_H_