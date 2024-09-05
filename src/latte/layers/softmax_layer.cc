#include "latte/layers/softmax_layer.h"
#include "latte/util/math_functions.h"

namespace latte {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                  const vector<Blob<Dtype> *> &top) {
  // N * C * H * W，由于axis=1，则此值为1
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);    // input_num * output_num
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));    // output_num
  sum_multiplier_.Reshape(mult_dims);                           // output_num
  Dtype *multiplier_data = sum_multiplier_.mutable_cpu_data();
  latte_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  outer_num_ = bottom[0]->count(0, softmax_axis_);   // n = input_num
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);  // h * w = 1
  vector<int> scale_dims = bottom[0]->shape();       // shape_
  // 将scale_dims的c设置为1，此时scale_dims和bottom[0]的c不同
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);  // 将scale_设置为n * 1 * h * w
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();  // 输入数据
  Dtype *top_data = top[0]->mutable_cpu_data();      // 输出数据
  Dtype *scale_data = scale_.mutable_cpu_data();     // 中间结果
  int channels = bottom[0]->shape(softmax_axis_);    // 通道数c
  int dim = bottom[0]->count() / outer_num_;         // 类别数目c * h * w
  // 将bottom数据copy到top中， 在reshape中，将top和bottom的shape设置成相同
  latte_copy(bottom[0]->count(), bottom_data, top_data);
  // 取出最大值，为了数值稳定, outer_num_ = n
  for (int i = 0; i < outer_num_; ++i) {
    latte_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; ++j) {
      // inner_num_ = h * w
      for (int k = 0; k < inner_num_; ++k) {
        // 获取每个维度的最大值
        scale_data[k] =
            std::max(scale_data[k], bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // 通过矩阵相乘的方式来计算
    // C = alpha * op(A) * op(B) + beta * C
    latte_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
                          -1., sum_multiplier_.cpu_data(), scale_data, 1.,
                          top_data);
    // 计算分子
    latte_exp<Dtype>(dim, top_data, top_data);
    // 计算分母
    latte_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1., top_data,
                          sum_multiplier_.cpu_data(), 0., scale_data);
    // 除法
    for (int j = 0; j < channels; ++j) {
      latte_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }
  }
}

#ifndef USE_CUDA
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);

}  // namespace latte
