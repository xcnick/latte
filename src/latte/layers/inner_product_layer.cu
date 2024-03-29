#include "latte/filler.h"
#include "latte/layers/inner_product_layer.h"
#include "latte/util/math_functions.h"

namespace latte {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->gpu_data();
  Dtype *top_data = top[0]->mutable_gpu_data();
  const Dtype *weight = this->blobs_[0]->gpu_data();
  if (M_ == 1) {
    latte_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1., weight, bottom_data,
                          (Dtype)0., top_data);
    if (bias_term_)
      latte_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    latte_gpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1., bottom_data, weight, (Dtype)0.,
                          top_data);
    if (bias_term_)
      latte_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace latte
