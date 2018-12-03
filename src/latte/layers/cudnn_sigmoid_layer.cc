#ifdef USE_CUDNN

#include "latte/layers/cudnn_sigmoid_layer.h"

namespace latte {

template <typename Dtype>
void CuDNNSigmoidLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  SigmoidLayer<Dtype>::LayerSetUp(bottom, top);
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  cudnn::createActivationDescriptor<Dtype>(&activ_desc_,
                                           CUDNN_ACTIVATION_SIGMOID);
  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNSigmoidLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  SigmoidLayer<Dtype>::Reshape(bottom, top);
  const int N = bottom[0]->shape(0);
  const int K = bottom[0]->shape(1);
  const int H = bottom[0]->shape(2);
  const int W = bottom[0]->shape(3);
  cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, N, K, H, W);
  cudnn::setTensor4dDesc<Dtype>(&top_desc_, N, K, H, W);
}

template <typename Dtype>
CuDNNSigmoidLayer<Dtype>::~CuDNNSigmoidLayer() {
  if (!handles_setup_) {
    return;
  }

  cudnnDestroyTensorDescriptor(this->bottom_desc_);
  cudnnDestroyTensorDescriptor(this->top_desc_);
  cudnnDestroy(this->handle_);
}

INSTANTIATE_CLASS(CuDNNSigmoidLayer);

template <typename Dtype>
void CuDNNSigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->gpu_data();
  Dtype *top_data = top[0]->mutable_gpu_data();
  CUDNN_CHECK(cudnnActivationForward(
      this->handle_, activ_desc_, cudnn::dataType<Dtype>::one,
      this->bottom_desc_, bottom_data, cudnn::dataType<Dtype>::zero,
      this->top_desc_, top_data));
}

template <typename Dtype>
void CuDNNSigmoidLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const Dtype *top_data = top[0]->gpu_data();
  const Dtype *top_diff = top[0]->gpu_diff();
  const Dtype *bottom_data = bottom[0]->gpu_data();
  Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
  CUDNN_CHECK(cudnnActivationBackward(
      this->handle_, activ_desc_, cudnn::dataType<Dtype>::one, this->top_desc_,
      top_data, this->top_desc_, top_diff, this->bottom_desc_, bottom_data,
      cudnn::dataType<Dtype>::zero, this->bottom_desc_, bottom_diff));
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNSigmoidLayer);

}  // namespace latte

#endif