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
    phase_ = param.phase();
    if (layer_param_.blobs_size() > 0) {
      blobs_.resize(layer_param_.blobs_size());
      for (int i = 0; i < layer_param_.blobs_size(); ++i) {
        blobs_[i].reset(new Blob<Dtype>());
        blobs_[i]->FromProto(layer_param_.blobs(i));
      }
    }
    for (int i = 0; i < layer_param_.param_size(); ++i) {
      if (layer_param_.param(i).lr_mult_oneof_case() ==
          ParamSpec::LrMultOneofCase::LR_MULT_ONEOF_NOT_SET) {
        layer_param_.mutable_param(i)->set_lr_mult(1.f);
      }
      if (layer_param_.param(i).decay_mult_oneof_case() ==
          ParamSpec::DecayMultOneofCase::DECAY_MULT_ONEOF_NOT_SET) {
        layer_param_.mutable_param(i)->set_decay_mult(1.f);
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
    SetLossWeights(top);           // 设置损失权值因子Blob
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

  // top: diff域来自上一层的误差梯度
  // propagate_down: 多路开关，与Bottom Blob维度相同，
  //     每个值表示是否将误差传递到Bottom Blob
  // bottom: 计算得到
  // 此函数会调用Backward_cpu/gpu，由派生类负责实现
  inline void Backward(const vector<Blob<Dtype> *> &top,
                       const vector<bool> &propagete_down,
                       const vector<Blob<Dtype> *> &bottom);

  // 内部可训练权重/偏置值
  vector<shared_ptr<Blob<Dtype>>> &blobs() { return blobs_; }

  // layer初始化参数，由Protobuf提供
  const LayerParameter &layer_param() const { return layer_param_; }

  // 将layer初始化参数写入Protobuff缓冲区
  virtual void ToProto(LayerParameter *param, bool write_diff = false);

  // 返回某个Top Blob相关的标量loss值
  inline Dtype loss(const int top_index) const {
    return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
  }

  // 设置与某个Top Blob相关的标量loss值
  inline void set_loss(const int top_index, const Dtype value) {
    if (loss_.size() <= top_index) {
      loss_.resize(top_index + 1, Dtype(0));
    }
    loss_[top_index] = value;
  }

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

  // 返回某些输入Blob允许强制反向传播
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

  // 指定该layer是否计算相对权值或偏置项的梯度，由param_id指定相对
  inline bool param_propagate_down(const int param_id) {
    return (param_propagate_down_.size() > param_id)
               ? param_propagate_down_[param_id]
               : false;
  }

  // 设置该层是否计算相对权值或偏置项的梯度，由param_id指定相对
  inline void set_param_propagate_down(const int param_id, const bool value) {
    if (param_propagate_down_.size() <= param_id) {
      param_propagate_down_.resize(param_id + 1, true);
    }
    param_propagate_down_[param_id] = value;
  }

 protected:
  // 存储layer参数的protobuf对象，属于LayerParameter对象
  LayerParameter layer_param_;
  // TRAIN or TEST
  Phase phase_;
  // 内部权值或偏置项，使用Blob存储
  vector<shared_ptr<Blob<Dtype>>> blobs_;
  // 标志位，是否计算对应参数的误差值
  vector<bool> param_propagate_down_;

  // 非零权重
  vector<Dtype> loss_;

  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top) = 0;

  virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top) {
    return Forward_cpu(bottom, top);
  }

  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom) = 0;

  virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom) {
    Backward_cpu(top, propagate_down, bottom);
  }

  // 检验输入输出的Blob数目是否满足layer的要求
  virtual void CheckBlobCounts(const vector<Blob<Dtype> *> &bottom,
                               const vector<Blob<Dtype> *> &top) {
    if (ExactNumBottomBlobs() >= 0) {
      CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
          << type() << " Layer takes" << ExactNumBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MinBottomBlobs() >= 0) {
      CHECK_LE(MinBottomBlobs(), bottom.size())
          << type() << " Layer takes at least " << MinBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MaxBottomBlobs() >= 0) {
      CHECK_GE(MaxBottomBlobs(), bottom.size())
          << type() << " Layer takes at least " << MaxBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (ExactNumTopBlobs() >= 0) {
      CHECK_EQ(ExactNumTopBlobs(), top.size())
          << type() << " Layer produces " << ExactNumTopBlobs()
          << " top blob(s) as output.";
    }
    if (MinTopBlobs() >= 0) {
      CHECK_LE(MinTopBlobs(), top.size())
          << type() << " Layer produces at least " << MinTopBlobs()
          << " top blob(s) as output.";
    }
    if (MaxTopBlobs() >= 0) {
      CHECK_GE(MaxTopBlobs(), top.size())
          << type() << " Layer produces at most " << MaxTopBlobs()
          << " top blob(s) as output.";
    }
    if (EqualNumBottomTopBlobs()) {
      CHECK_EQ(bottom.size(), top.size())
          << type() << " Layer produces one top blob as output for each "
          << "bottom blob input.";
    }
  }

  // 在SetUp函数中被调用，初始化Top Blob相关的loss权重，非零权重放到Top
  // Blob的diff中
  inline void SetLossWeights(const vector<Blob<Dtype> *> &top) {
    const int num_loss_weights = layer_param_.loss_weight_size();
    if (num_loss_weights) {
      CHECK_EQ(top.size(), num_loss_weights)
          << "loss_weight must be "
             "unspecified or specified once per top blob.";
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        const Dtype loss_weight = layer_param_.loss_weight(top_id);
        if (loss_weight == Dtype(0)) {
          continue;
        }
        this->set_loss(top_id, loss_weight);
        const int count = top[top_id]->count();
        Dtype *loss_multiplier = top[top_id]->mutable_cpu_diff();
        latte_set(count, loss_weight, loss_multiplier);
      }
    }
  }
};

template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {
  Dtype loss = 0;
  Reshape(bottom, top);
  switch (Latte::mode()) {
    case Latte::CPU:
      Forward_cpu(bottom, top);
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        if (!this->loss(top_id)) {
          continue;
        }
        const int count = top[top_id]->count();
        const Dtype *data = top[top_id]->cpu_data();
        const Dtype *loss_weights = top[top_id]->cpu_diff();
        loss += latte_cpu_dot(count, data, loss_weights);
      }
      break;
    case Latte::GPU:
      Forward_gpu(bottom, top);
#ifdef WITH_CUDA
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        if (!this->loss(top_id)) {
          continue;
        }
        const int count = top[top_id]->count();
        const Dtype *data = top[top_id]->gpu_data();
        const Dtype *loss_weights = top[top_id]->gpu_diff();
        Dtype blob_loss = 0;
        latte_gpu_dot(count, data, loss_weights, &blob_loss);
        loss += blob_loss;
      }
#endif
      break;
    default:
      LOG(FATAL) << "Unknown latte mode.";
  }
  return loss;
}

template <typename Dtype>
inline void Layer<Dtype>::Backward(const vector<Blob<Dtype> *> &top,
                                   const vector<bool> &propagate_down,
                                   const vector<Blob<Dtype> *> &bottom) {
  switch (Latte::mode()) {
    case Latte::CPU:
      Backward_cpu(top, propagate_down, bottom);
      break;
    case Latte::GPU:
      Backward_gpu(top, propagate_down, bottom);
      break;
    default:
      LOG(FATAL) << "Unknown latte mode.";
  }
}

template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter *param, bool write_diff) {
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
  for (int i = 0; i < blobs_.size(); ++i) {
    blobs_[i]->ToProto(param->add_blobs(), write_diff);
  }
}

}  // namespace latte

#endif  // LATTE_LAYER_H_