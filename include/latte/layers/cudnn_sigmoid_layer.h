#ifndef LATTE_CUDNN_SIGMOID_LAYER_H_
#define LATTE_CUDNN_SIGMOID_LAYER_H_

#include "latte/layers/sigmoid_layer.h"

namespace latte {

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNSigmoidLayer : public SigmoidLayer<Dtype> {
 public:
  explicit CuDNNSigmoidLayer(const LayerParameter &param)
      : SigmoidLayer<Dtype>(param), handles_setup_(false) {}

  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top) override;
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top) override;

  virtual ~CuDNNSigmoidLayer();

 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top) override;

  bool handles_setup_;
  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t bottom_desc_;
  cudnnTensorDescriptor_t top_desc_;
  cudnnActivationDescriptor_t activ_desc_;
};

#endif
}  // namespace latte

#endif  // LATTE_CUDNN_SIGMOID_LAYER_H_