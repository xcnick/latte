#ifndef LATTE_NET_H_
#define LATTE_NET_H_

#include "latte/blob.h"
#include "latte/common.h"
#include "latte/layer.h"
#include "latte/proto/latte.pb.h"

namespace latte {

template <typename Dtype>
class Net : public Noncopyable {
 public:
  explicit Net(const NetParameter &param);

  // 使用prototxt文件来传入网络结构参数, 一般用于开发测试
  // phase：当前net是进行TEST还是TRAIN
  // level：指出当前net中哪些layer要包含在net中
  // stage：指出当前net中哪些layer要包含在net中
  explicit Net(const string &param_file, const int level = 0,
               const vector<string> *stages = nullptr);
  virtual ~Net() {}

  // 使用NetParameter初始化
  // 对net中各个layer，每个layer输入输出blob，layer初始化
  void Init(const NetParameter &param);

  const vector<Blob<Dtype> *> &Forward();

  Dtype ForwardFromTo(int start, int end);
  Dtype ForwardFrom(int start);
  Dtype ForwardTo(int end);

  // 调整Layers的shape
  void Reshape();

  // 共享权值和偏置数据，仅在Net::Init中被调用
  void ShareWeights();

  // 从另一个Net拷贝pre-trained layers
  void ShareTrainedLayersWith(const Net *other);

  // 从另一个Net拷贝pre-trained layers，通过NetParameter对象
  void CopyTrainedLayersFrom(const NetParameter &param);
  void CopyTrainedLayersFrom(const string &trained_filename);
  void CopyTrainedLayersFromBinaryProto(const string &trained_filename);

  // 将Net写入到Proto文件
  void ToProto(NetParameter *param) const;

  // Net名称
  inline const string &name() const { return name_; }
  // 所有Layer名称
  inline const vector<string> &layer_names() const { return layer_names_; }
  // 所有Blob名称
  inline const vector<string> &blob_names() const { return blob_names_; }
  // 所有Blob
  inline const vector<shared_ptr<Blob<Dtype>>> &blobs() const { return blobs_; }
  // 所有Layer
  inline const vector<shared_ptr<Layer<Dtype>>> &layers() const {
    return layers_;
  }

  inline const vector<vector<Blob<Dtype> *>> &bottom_vecs() const {
    return bottom_vecs_;
  }

  inline const vector<vector<Blob<Dtype> *>> &top_vecs() const {
    return top_vecs_;
  }

  inline const vector<int> &top_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, static_cast<int>(top_id_vecs_.size())) << "Invalid layer id";
    return top_id_vecs_[i];
  }

  inline const vector<int> &bottom_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, static_cast<int>(bottom_id_vecs_.size())) << "Invalid layer id";
    return bottom_id_vecs_[i];
  }

  inline const vector<shared_ptr<Blob<Dtype>>> &params() const {
    return params_;
  }

  inline const map<string, int> &param_names_index() const {
    return param_names_index_;
  }

  inline const vector<int> &param_owners() const { return param_owners_; }

  inline const vector<string> &param_display_names() const {
    return param_display_names_;
  }

  inline const int num_inputs() const { return net_input_blobs_.size(); }
  inline const int num_outputs() const { return net_output_blobs_.size(); }

  inline const vector<Blob<Dtype> *> &input_blobs() const {
    return net_input_blobs_;
  }

  inline const vector<Blob<Dtype> *> &output_blobs() const {
    return net_output_blobs_;
  }

  inline const vector<int> &input_blob_indices() const {
    return net_input_blob_indices_;
  }

  inline const vector<int> &output_blob_indices() const {
    return net_output_blob_indices_;
  }

  bool has_blob(const string &blob_name) const;
  const shared_ptr<Blob<Dtype>> blob_by_name(const string &blob_name) const;
  bool has_layer(const string &layer_name) const;
  const shared_ptr<Layer<Dtype>> layer_by_name(const string &layer_name) const;

  void set_debug_info(const bool value) { debug_info_ = value; }

  // 移除指定layers
  static void FilterNet(const NetParameter &param,
                        NetParameter *param_filtered);
  // 检查NetState是否匹配NetStateRule
  static bool StateMeetsRule(const NetState &state, const NetStateRule &rule,
                             const string &layer_name);

  class Callback {
   protected:
    virtual void run(int layer) = 0;

    template <typename T>
    friend class Net;
  };

  const vector<Callback *> &before_forward() const { return before_forward_; }
  void add_before_forward(Callback *value) { before_forward_.push_back(value); }
  const vector<Callback *> &after_forward() const { return after_forward_; }
  void add_after_forward(Callback *value) { after_forward_.push_back(value); }

 protected:
  // 给某层增加一个top blob
  void AppendTop(const NetParameter &param, const int layer_id,
                 const int top_id, set<string> *available_blobs,
                 map<string, int> *blob_name_to_idx);
  // 给某层增加一个bottom blob
  int AppendBottom(const NetParameter &param, const int layer_id,
                   const int bottom_id, set<string> *available_blobs,
                   map<string, int> *blob_name_to_idx);

  void ForwardDebugInfo(const int layer_id);

  // 网络名称
  string name_;
  // layers
  vector<shared_ptr<Layer<Dtype>>> layers_;
  vector<string> layer_names_;
  map<string, int> layer_names_index_;

  // 存储每个layer的中间结果
  vector<shared_ptr<Blob<Dtype>>> blobs_;
  vector<string> blob_names_;
  map<string, int> blob_names_index_;

  // 存储每个layer bottom blobs指针
  vector<vector<Blob<Dtype> *>> bottom_vecs_;
  vector<vector<int>> bottom_id_vecs_;

  // 存储每个layer top blobs指针
  vector<vector<Blob<Dtype> *>> top_vecs_;
  vector<vector<int>> top_id_vecs_;

  // layer的loss函数值
  vector<vector<int>> param_id_vecs_;
  vector<int> param_owners_;
  vector<string> param_display_names_;
  vector<pair<int, int>> param_layer_indices_;
  map<string, int> param_names_index_;

  // blob index
  vector<int> net_input_blob_indices_;
  vector<int> net_output_blob_indices_;
  vector<Blob<Dtype> *> net_input_blobs_;
  vector<Blob<Dtype> *> net_output_blobs_;
  // parameters
  vector<shared_ptr<Blob<Dtype>>> params_;
  /**
   * The mapping from params_ -> learnable_params_: we have
   * learnable_param_ids_.size() == params_.size(),
   * and learnable_params_[learnable_param_ids_[i]] == params_[i].get()
   * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
   * and learnable_params_[learnable_param_ids_[i]] gives its owner.
   */
  // Net使用的内存字节数
  size_t memory_used_;
  // 是否显示debug信息
  bool debug_info_;
  // Callbacks
  vector<Callback *> before_forward_;
  vector<Callback *> after_forward_;
};
}  // namespace latte

#endif  // LATTE_NET_H_