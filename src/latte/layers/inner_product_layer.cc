#include "latte/layers/inner_product_layer.h"
#include "latte/filler.h"
#include "latte/util/math_functions.h"

namespace latte {

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  if (this->layer_param_.inner_product_param().has_bias_term()) {
    bias_term_ = this->layer_param_.inner_product_param().bias_term().value();
  } else {
    bias_term_ = true;
  }
  int axis;
  if (this->layer_param_.inner_product_param().has_axis()) {
    axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis().value());
  } else {
    axis = 1;
  }
  if (this->layer_param_.inner_product_param().has_transpose()) {
    transpose_ = this->layer_param_.inner_product_param().transpose().value();
  } else {
    transpose_ = false;
  }

  N_ = num_output;
  // K_ = C * H * W, 表示单个样本的特征长度
  K_ = bottom[0]->count(axis);
  // 初始化weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    // 如果包含偏置项，则将blobs_的size设置为2
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // 设置weights的维度
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // 初始化赋值weights
    shared_ptr<Filler<Dtype>> weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // 如果包含偏置项，则初始化偏置项
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype>> bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  // 确认维度
  // const int axis = bottom[0]->CanonicalAxisIndex(
  //     this->layer_param_.inner_product_param().axis());
  int axis;
  if (this->layer_param_.inner_product_param().has_axis()) {
    axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis().value());
  } else {
    axis = 1;
  }
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // batch_size
  M_ = bottom[0]->count(0, axis);
  // 将输出项top[0]的维度置为 batch_size * N_
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // 设置偏置项乘子
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    // 将偏置项乘子初始化为1
    latte_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const Dtype *weight = this->blobs_[0]->cpu_data();
  // top_data (M * N) = bottom_data (M * K) * weight' (K * N)
  latte_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
                        M_, N_, K_, (Dtype)1., bottom_data, weight, (Dtype)0.,
                        top_data);
  if (bias_term_) {
    // top_data (M * N) += bias_multi (M * 1) * bias(1 * N)
    latte_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                          bias_multiplier_.cpu_data(),
                          this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype *top_diff = top[0]->cpu_diff();
    const Dtype *bottom_data = bottom[0]->cpu_data();
    // weight_diff (N * K) += top_diff' (N * M) * bottom_data (M * K)
    if (transpose_) {
      latte_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype)1.,
                            bottom_data, top_diff, (Dtype)1.,
                            this->blobs_[0]->mutable_cpu_diff());
    } else {
      latte_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
                            top_diff, bottom_data, (Dtype)1.,
                            this->blobs_[0]->mutable_cpu_diff());
    }

    if (bias_term_ && this->param_propagate_down_[1]) {
      const Dtype *top_diff = top[0]->cpu_diff();
      // bias_diff (N * 1) = top_diff' (N * M) * bias_multi (M * 1)
      latte_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
                            bias_multiplier_.cpu_data(), (Dtype)1.,
                            this->blobs_[1]->mutable_cpu_diff());
    }
    if (propagate_down[0]) {
      const Dtype *top_diff = top[0]->cpu_diff();
      // bottom_diff (M * K) = top_diff (M * N) * weight (N * K)
      if (transpose_) {
        latte_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype)1.,
                              top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
                              bottom[0]->mutable_cpu_diff());
      } else {
        latte_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
                              top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
                              bottom[0]->mutable_cpu_diff());
      }
    }
  }
}

#ifndef WITH_CUDA
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace latte