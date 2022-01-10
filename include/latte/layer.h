#ifndef LATTE_LAYER_H_
#define LATTE_LAYER_H_

#include "latte/blob.h"
#include "latte/common.h"
#include "latte/layer_factory.h"
#include "latte/proto/latte.pb.h"
#include "latte/util/math_functions.h"

namespace latte {

template <typename Dtype>
class Layer : public Noncopyable {
 public:
  explicit Layer(const LayerParameter &param) : layer_param_(param) {
    if (layer_param_.blobs_size() > 0) {
      blobs_.resize(layer_param_.blobs_size());
      for (int i = 0; i < layer_param_.blobs_size(); ++i) {
        blobs_[i].reset(new Blob<Dtype>());
        blobs_[i]->FromProto(layer_param_.blobs(i));
      }
    }
  }

  virtual ~Layer() = default;

  // 配置函数，不可被覆盖
  void SetUp(const vector<Blob<Dtype> *> &bottom,
             const vector<Blob<Dtype> *> &top) {
    CheckBlobCounts(bottom, top);  // 检查Blob
    LayerSetUp(bottom, top);       // 与曾类型相关的配置过程
    Reshape(bottom, top);          // 对Top Blob变形
  }

  // layer配置函数，特定类型的相关配置，由具体类型曾自己实现
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top) {}

  // 纯虚函数，修改Top Blob以及内部Blob缓冲区形状
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top) = 0;

  // 前向传播，给定Bottom Blob，计算Top Blob和loss，返回当前层loss
  inline Dtype Forward(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  // 内部可训练权重/偏置值
  vector<shared_ptr<Blob<Dtype>>> &blobs() { return blobs_; }

  // layer初始化参数，由Protobuf提供
  const LayerParameter &layer_param() const { return layer_param_; }

  // 将layer初始化参数写入Protobuff缓冲区
  virtual void ToProto(LayerParameter *param);

  // layer类型字符串
  virtual inline const char *type() const { return ""; }

  // 返回该层需要的输入Blob 数目,-1 表示不关心.
  virtual inline int ExactNumBottomBlobs() const { return -1; }

  // 需要输入Blob的最小数目
  virtual inline int MinBottomBlobs() const { return -1; }

  // 需要输入Blob的最大的数目
  virtual inline int MaxBottomBlobs() const { return -1; }

  // 需要输出Blob数目
  virtual inline int ExactNumTopBlobs() const { return -1; }

  // 需要输出Blob的最小数目
  virtual inline int MinTopBlobs() const { return -1; }

  // 需要输出Blob的最大的数目
  virtual inline int MaxTopBlobs() const { return -1; }

  // 返回该lyaer是否有相同的输入输出Blob
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }

  // 返回是否允许匿名Top Blob，由该layer自动创建足够多的Blob
  virtual inline bool AutoTopBlobs() const { return false; }

 protected:
  // 存储layer参数的protobuf对象，属于LayerParameter对象
  LayerParameter layer_param_;
  // 内部权值或偏置项，使用Blob存储
  vector<shared_ptr<Blob<Dtype>>> blobs_;

  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top) = 0;

  virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top) {
    return Forward_cpu(bottom, top);
  }

  // 检验输入输出的Blob数目是否满足layer的要求
  virtual void CheckBlobCounts(const vector<Blob<Dtype> *> &bottom,
                               const vector<Blob<Dtype> *> &top) {
    if (ExactNumBottomBlobs() >= 0) {
      CHECK_EQ(ExactNumBottomBlobs(), static_cast<int>(bottom.size()))
          << type() << " Layer takes" << ExactNumBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MinBottomBlobs() >= 0) {
      CHECK_LE(MinBottomBlobs(), static_cast<int>(bottom.size()))
          << type() << " Layer takes at least " << MinBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MaxBottomBlobs() >= 0) {
      CHECK_GE(MaxBottomBlobs(), static_cast<int>(bottom.size()))
          << type() << " Layer takes at least " << MaxBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (ExactNumTopBlobs() >= 0) {
      CHECK_EQ(ExactNumTopBlobs(), static_cast<int>(top.size()))
          << type() << " Layer produces " << ExactNumTopBlobs()
          << " top blob(s) as output.";
    }
    if (MinTopBlobs() >= 0) {
      CHECK_LE(MinTopBlobs(), static_cast<int>(top.size()))
          << type() << " Layer produces at least " << MinTopBlobs()
          << " top blob(s) as output.";
    }
    if (MaxTopBlobs() >= 0) {
      CHECK_GE(MaxTopBlobs(), static_cast<int>(top.size()))
          << type() << " Layer produces at most " << MaxTopBlobs()
          << " top blob(s) as output.";
    }
    if (EqualNumBottomTopBlobs()) {
      CHECK_EQ(bottom.size(), top.size())
          << type() << " Layer produces one top blob as output for each "
          << "bottom blob input.";
    }
  }
};

template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {
  Reshape(bottom, top);
  switch (Latte::mode()) {
    case Latte::CPU:
      Forward_cpu(bottom, top);
      break;
    case Latte::GPU:
      Forward_gpu(bottom, top);
      break;
    default:
      LOG(FATAL) << "Unknown latte mode.";
  }
  return 0;
}

template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter *param) {
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
  for (size_t i = 0; i < blobs_.size(); ++i) {
    blobs_[i]->ToProto(param->add_blobs());
  }
}

}  // namespace latte

#endif  // LATTE_LAYER_H_