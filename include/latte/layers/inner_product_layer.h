#ifndef LATTE_INNER_PRODUCT_LAYER_H_
#define LATTE_INNER_PRODUCT_LAYER_H_

#include "latte/blob.h"
#include "latte/layer.h"
#include "latte/proto/latte.pb.h"

namespace latte {

template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
 public:
  explicit InnerProductLayer(const LayerParameter &param)
      : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "InnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);

  // 矩阵乘法参数(M, K) * (K, N) = (M, N)
  int M_;  // 输入数据的数量, 即batch_size
  int K_;  // 单个输入数据包含的元素个数 C * H * W，特征维度
  int N_;  // 输出神经元个数
  // 是否包含偏置项
  bool bias_term_;
  // 偏置项乘子
  Blob<Dtype> bias_multiplier_;
  // 是否转置权重矩阵
  bool transpose_;
};

}  // namespace latte

#endif  // LATTE_INNER_PRODUCT_LAYER_H_