#ifdef USE_CUDNN

#include "latte/layers/cudnn_softmax_layer.h"

namespace latte {

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  SoftmaxLayer<Dtype>::LayerSetUp(bottom, top);
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  SoftmaxLayer<Dtype>::Reshape(bottom, top);
  int N = this->outer_num_;
  int K = bottom[0]->shape(this->softmax_axis_);
  int H = this->inner_num_;
  int W = 1;
  cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, N, K, H, W);
  cudnn::setTensor4dDesc<Dtype>(&top_desc_, N, K, H, W);
}

template <typename Dtype>
CuDNNSoftmaxLayer<Dtype>::~CuDNNSoftmaxLayer() {
  if (!handles_setup_) {
    return;
  }
  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroy(handle_);
}

INSTANTIATE_CLASS(CuDNNSoftmaxLayer);

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->gpu_data();
  Dtype *top_data = top[0]->mutable_gpu_data();
  CUDNN_CHECK(cudnnSoftmaxForward(
      handle_, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
      cudnn::dataType<Dtype>::one, bottom_desc_, bottom_data,
      cudnn::dataType<Dtype>::zero, top_desc_, top_data));
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNSoftmaxLayer);

}  // namespace latte

#endif