#include <cmath>

#include "latte/layers/sigmoid_layer.h"

namespace latte {

template <typename Dtype>
__global__ void SigmoidForward(const int n, const Dtype *in, Dtype *out) {
  CUDA_KERNEL_LOOP(index, n) { out[index] = 1. / (1. + exp(-in[index])); }
}

template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->gpu_data();
  Dtype *top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  SigmoidForward<Dtype><<<LATTE_GET_BLOCKS(count), LATTE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidLayer);

}  // namespace latte