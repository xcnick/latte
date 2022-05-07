#include "latte/net.h"

#include "latte/common.h"
#include "latte/proto/latte.pb.h"
#include "latte/util/insert_splits.h"
#include "latte/util/io.h"

namespace latte {

// 一般在训练阶段由Solver调用
template <typename Dtype>
Net<Dtype>::Net(const NetParameter &param) {
  Init(param);
}

template <typename Dtype>
Net<Dtype>::Net(const string &param_file, const int level,
                const vector<string> *stages) {
  NetParameter param;
  ReadProtoFromTextFileOrDie(param_file, &param);
  if (stages != nullptr) {
    for (size_t i = 0; i < stages->size(); ++i) {
      param.mutable_state()->add_stage((*stages)[i]);
    }
  }
  param.mutable_state()->set_level(level);
  Init(param);
}

template <typename Dtype>
void Net<Dtype>::Init(const NetParameter &in_param) {
  // 使用stages/level规则对NetParameter进行过滤, 过滤后放入filtered_param
  NetParameter filtered_param;
  FilterNet(in_param, &filtered_param);
  LOG(INFO) << "Initializing net from parameters: " << std::endl
            << filtered_param.DebugString();
  // 对过滤后的filtered_param拷贝一个副本
  NetParameter param;
  InsertSplits(filtered_param, &param);
  // 创建layers，并创建相互之间的连接
  name_ = param.name();
  map<string, int> blob_name_to_idx;
  set<string> available_blobs;
  memory_used_ = 0;
  // 各个层的输入blob，二维
  bottom_vecs_.resize(param.layer_size());
  // 各个层的输出blob
  top_vecs_.resize(param.layer_size());
  // 各个层输入blob的id
  bottom_id_vecs_.resize(param.layer_size());
  // 各个层的参数blob的id
  param_id_vecs_.resize(param.layer_size());
  // 各个层输出blob的id
  top_id_vecs_.resize(param.layer_size());
  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    const LayerParameter &layer_param = param.layer(layer_id);
    layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
    layer_names_.push_back(layer_param.name());
    LOG(INFO) << "Creating Layer " << layer_param.name();

    // 1.初始化bottom blob: 将bottom_vecs_的地址与blobs_[blob_id]地址关联起来,
    // 将bottom_id_vecs_与blob_id_关联起来;
    // 2.对于数据输入层来说只有top,没有bottom,所以会跳过下面的for循环
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size();
         ++bottom_id) {
      // 1.net中bottom/top是交替初始化的,前一层的top是后一层的bottom,前一层top的
      // available_blobs/blob_name_to_idx参数就是后一层的bottom参数
      // 2.AppendBottom将bottom_vecs_与blobs_[id]关联起来, 将bottom_id_vecs_与
      // blob_id_关联起来
      AppendBottom(param, layer_id, bottom_id, &available_blobs,
                   &blob_name_to_idx);
    }
    int num_top = layer_param.top_size();
    // 初始化top blob: 将top_vecs_的地址与blobs_[blob_id]地址关联起来,
    // 将top_id_vecs_与blob_id_关联起来; AppendTop还创建了新blob
    for (int top_id = 0; top_id < num_top; ++top_id) {
      // 通过AppendTop和AppendBottom, bottom_vecs_和top_vecs_连接在了一起
      // 在AppendTop中会往available_blobs添加某层的输出blob,在AppendBottom中会
      // 从available_blobs中删除前一层的输出blob，所有layers遍历完后剩下的就
      // 是整个net的输出blob
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
      // 对于整个net的输入层，每通过AppendTop新建一个top blob, blobs.size()
      // 就增加1,blobs_size()是从0开始增加的，就能代表整个net输入blob的id
      if (layer_param.type() == "Input") {
        const int blob_id = blobs_.size() - 1;
        net_input_blob_indices_.push_back(blob_id);
        net_input_blobs_.push_back(blobs_[blob_id].get());
      }
    }
    // 补齐top blob，使该层的top blob个数达到要求
    Layer<Dtype> *layer = layers_[layer_id].get();
    if (layer->AutoTopBlobs()) {
      const int needed_num_top =
          std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
      for (; num_top < needed_num_top; ++num_top) {
        AppendTop(param, layer_id, num_top, nullptr, nullptr);
      }
    }
    layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    LOG(INFO) << "Setting up " << layer_names_[layer_id];

    LOG(INFO) << "Memory required for data: " << memory_used_ * sizeof(Dtype);

    // 对参数进行初始化：一般权值weight存放在一个blob,偏置bias存放在另一个blob
    // 本层的param_need_backward(具体值来自LayerParameter)和本层的
    // blob_need_backward_决定了本层的need_backward；本层的need_backward决
    // 定了本层的layer_need_backward_

    // LayerParameter中已经定义了的参数个数(可能小于实际的个数)
    // const int param_size = layer_param.param_size();
    // // 实际的参数
    // const int num_param_blobs = layers_[layer_id]->blobs().size();
    // CHECK_LE(param_size, num_param_blobs)
    //     << "Too many params specified for layer " << layer_param.name();

    // 一个layer一般有两个参数Blob, 第一个存weight, 第二个存bias
    // for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
    //   AppendParam(param, layer_id, param_id);
    // }
  }

  // In the end, all remaining blobs are considered output blobs.
  // 在AppendBottom中已经将bottom blob从available_blobs中删掉,最终只剩下最顶
  // 层的top blob,就是输出blob
  for (auto it = available_blobs.begin(); it != available_blobs.end(); ++it) {
    LOG(INFO) << "This network produces output " << *it;
    net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
    net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
  }
  for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;
  }
  for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;
  }
  ShareWeights();
  debug_info_ = param.debug_info().value();
  LOG(INFO) << "Network initialization done.";
}

template <typename Dtype>
void Net<Dtype>::FilterNet(const NetParameter &param,
                           NetParameter *param_filtered) {
  // 在定义net结构的prototxt文件中往往会定义某层的include/exclude参数,
  // include表示如果在构造net时如果满足include的条件，本层就包含在net中；
  // exclude表示在构造net时如果满足exclude条件，本层就不会包含在net中。
  // prototxt的这个include/exclude参数被读取后就是caffe.proto中的NetStateRule类，
  // 类中有phase、min_level、max_level、stage、not_stage 5个参数
  NetState net_state(param.state());
  param_filtered->CopyFrom(param);
  // 先清除layers,然后根据规则重新添加layers
  param_filtered->clear_layer();
  for (int i = 0; i < param.layer_size(); ++i) {
    const LayerParameter &layer_param = param.layer(i);
    const string &layer_name = layer_param.name();
    // include和exclude不能同时存在
    CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
        << "Specify either include rules or exclude rules; not both.";
    // If no include rules are specified, the layer is included by default and
    // only excluded if it meets one of the exclude rules.
    // 对于包含include参数的层：如果满足min_level<level<max_level 或
    // stages中任意一个元素能在NetStateRule::stage中找到, 该层就会被保留在net中
    // 对于包含exclude参数的层：如果满足min_level<level<max_level  或
    // stages中任意一个元素能在NetStateRule::stage中找到, 该层就会从net中剔除
    // 当然如果是在NetStateRule::not_stage中找到， 结果正好相反
    bool layer_included = (layer_param.include_size() == 0);
    for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {
        layer_included = false;
      }
    }
    for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
        layer_included = true;
      }
    }
    if (layer_included) {
      param_filtered->add_layer()->CopyFrom(layer_param);
    }
  }
}

template <typename Dtype>
bool Net<Dtype>::StateMeetsRule(const NetState &state, const NetStateRule &rule,
                                const string &layer_name) {
  // Check whether the rule is broken due to min level.
  if (rule.has_min_level()) {
    if (state.level() < rule.min_level().value()) {
      LOG(INFO) << "The NetState level (" << state.level()
                << ") is above the min_level (" << rule.min_level().value()
                << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to max level.
  if (rule.has_max_level()) {
    if (state.level() > rule.max_level().value()) {
      LOG(INFO) << "The NetState level (" << state.level()
                << ") is above the max_level (" << rule.max_level().value()
                << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to stage. The NetState must
  // contain ALL of the rule's stages to meet it.
  for (int i = 0; i < rule.stage_size(); ++i) {
    // Check that the NetState contains the rule's ith stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.stage(i) == state.stage(j)) {
        has_stage = true;
      }
    }
    if (!has_stage) {
      LOG(INFO) << "The NetState did not contain stage '" << rule.stage(i)
                << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to not_stage. The NetState must
  // contain NONE of the rule's not_stages to meet it.
  for (int i = 0; i < rule.not_stage_size(); ++i) {
    // Check that the NetState contains the rule's ith not_stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.not_stage(i) == state.stage(j)) {
        has_stage = true;
      }
    }
    if (has_stage) {
      LOG(INFO) << "The NetState contained a not_stage '" << rule.not_stage(i)
                << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  return true;
}

// AppendTop函数会向整个net的blob列表（blobs_）中添加一个新blob，同时将本层新建的top
// blob指向该新增blob， 这样就把层的输出blob和blob列表(blobs_)关联起来了
template <typename Dtype>
void Net<Dtype>::AppendTop(const NetParameter &param, const int layer_id,
                           const int top_id, set<string> *available_blobs,
                           map<string, int> *blob_name_to_idx) {
  shared_ptr<LayerParameter> layer_param(
      new LayerParameter(param.layer(layer_id)));
  const string &blob_name = (layer_param->top_size() > top_id)
                                ? layer_param->top(top_id)
                                : "(automatic)";
  // Check if we are doing in-place computation
  // 同址计算:top blob使用和bottom blob相同的地址和id
  // 是否使用同址计算由prototxt中对top/bottom blob名字的定义决定
  if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
      blob_name == layer_param->bottom(top_id)) {
    // In-place computation
    LOG(INFO) << layer_param->name() << " -> " << blob_name << " (in-place)";
    top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
    top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
  } else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    // blob_name_to_idx中的元素始终是在AppendTop中添加的,所以如果有重复名字,
    // 就意味之前有其他top blob同名
    LOG(FATAL) << "Top blob '" << blob_name
               << "' produced by multiple sources.";
  } else {
    // Normal output.
    // 不进行同址计算, top使用和bottom独立的Blob
    LOG(INFO) << layer_param->name() << " -> " << blob_name;

    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
    // 当前blob的个数, 就是要新增的blob的id(在 blobs_尾部新增一个blob
    const int blob_id = blobs_.size();
    // 新增一个blob的动作是在AppendTop中完成的,
    // AppendBottom中只是把当前层bottom和
    // 前一层top的地址关联起来(通过bottom/top指向相同的blobs_[id]/blob_id来连接)
    blobs_.push_back(blob_pointer);
    blob_names_.push_back(blob_name);
    // 新增blob_name_to_idx的键值, 对于数据输入层(只有top,没有bottom, 是第一层),
    // blob_name_to_idx也不是null,所以也会进入下面分支
    if (blob_name_to_idx) {
      (*blob_name_to_idx)[blob_name] = blob_id;
    }
    top_id_vecs_[layer_id].push_back(blob_id);
    top_vecs_[layer_id].push_back(blob_pointer.get());
  }
  // 在AppendTop中增加blob,在AppendBottom中剔除blob,遍历所有层后剩下的就是net的输出blob
  if (available_blobs) {
    available_blobs->insert(blob_name);
  }
}

// 给某层增加一个bottom blob
// 将bottom_vecs_与blobs_[id]关联起来, 将bottom_id_vecs_与blob_id_关联起来
template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter &param, const int layer_id,
                             const int bottom_id, set<string> *available_blobs,
                             map<string, int> *blob_name_to_idx) {
  const LayerParameter &layer_param = param.layer(layer_id);
  const string &blob_name = layer_param.bottom(bottom_id);
  if (available_blobs->find(blob_name) == available_blobs->end()) {
    LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
               << layer_param.name() << "', bottom index " << bottom_id << ")";
  }
  const int blob_id = (*blob_name_to_idx)[blob_name];
  LOG(INFO) << layer_names_[layer_id] << " <- " << blob_name;
  // 新增一个blob的动作是在top中完成的, bottom中只是把当前层bottom和前一层
  // top的地址连接起来(通过bottom/top指向相同的blobs_[id]/blob_id来连接)
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
  bottom_id_vecs_[layer_id].push_back(blob_id);
  // 上一层的AppendTop时insert入available_blob, 本层的AppendBottom时erase
  available_blobs->erase(blob_name);
  return blob_id;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, static_cast<int>(layers_.size()));
  Dtype loss = 0;
  for (int i = start; i <= end; ++i) {
    for (size_t c = 0; c < before_forward_.size(); ++c) {
      before_forward_[c]->run(i);
    }
    if (debug_info_) {
      ForwardDebugInfo(i);
    }
    for (size_t c = 0; c < after_forward_.size(); ++c) {
      after_forward_[c]->run(i);
    }
  }
  return loss;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFrom(int start) {
  return ForwardFromTo(start, layers_.size() - 1);
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardTo(int end) {
  return ForwardFromTo(0, end);
}

template <typename Dtype>
const vector<Blob<Dtype> *> &Net<Dtype>::Forward() {
  ForwardFromTo(0, layers_.size() - 1);
  return net_output_blobs_;
}

template <typename Dtype>
void Net<Dtype>::ForwardDebugInfo(const int layer_id) {
  for (size_t top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
    const Blob<Dtype> &blob = *top_vecs_[layer_id][top_id];
    const string &blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG(INFO) << "    [Forward] "
              << "Layer " << layer_names_[layer_id] << ", top blob "
              << blob_name << " data: " << data_abs_val_mean;
  }
  for (size_t param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    const Blob<Dtype> &blob = *layers_[layer_id]->blobs()[param_id];
    const int net_param_id = param_id_vecs_[layer_id][param_id];
    const string &blob_name = param_display_names_[net_param_id];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG(INFO) << "    [Forward] "
              << "Layer " << layer_names_[layer_id] << ", param blob "
              << blob_name << " data: " << data_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::ShareTrainedLayersWith(const Net *other) {
  int num_source_layers = other->layers().size();
  for (int i = 0; i < num_source_layers; ++i) {
    Layer<Dtype> *source_layer = other->layers()[i].get();
    const string &source_layer_name = other->layer_names()[i];
    int target_layer_id = 0;
    while (target_layer_id != static_cast<int>(layer_names_.size()) &&
           layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == static_cast<int>(layer_names_.size())) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > > &target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (size_t j = 0; j < target_blobs.size(); ++j) {
      Blob<Dtype> *source_blob = source_layer->blobs()[j].get();
      CHECK(target_blobs[j]->shape() == source_blob->shape())
          << "Cannot share param " << j << " weights from layer '"
          << source_layer_name << "'; shape mismatch.  Source param shape is "
          << source_blob->shape_string() << "; target param shape is "
          << target_blobs[j]->shape_string();
      target_blobs[j]->ShareData(*source_blob);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::Reshape() {
  for (size_t i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter &param) {
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter &source_layer = param.layer(i);
    const string &source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != static_cast<int>(layer_names_.size()) &&
           layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == static_cast<int>(layer_names_.size())) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > > &target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(static_cast<int>(target_blobs.size()), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (size_t j = 0; j < target_blobs.size(); ++j) {
      if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
        Blob<Dtype> source_blob;
        const bool kReshape = true;
        source_blob.FromProto(source_layer.blobs(j), kReshape);
        LOG(FATAL)
            << "Cannot copy param " << j << " weights from layer '"
            << source_layer_name << "'; shape mismatch.  Source param shape is "
            << source_blob.shape_string() << "; target param shape is "
            << target_blobs[j]->shape_string() << ". "
            << "To learn this layer's parameters from scratch rather than "
            << "copying from a saved net, rename the layer.";
      }
      const bool kReshape = false;
      target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string &trained_filename) {
  CopyTrainedLayersFromBinaryProto(trained_filename);
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromBinaryProto(
    const string &trained_filename) {
  NetParameter param;
  ReadProtoFromTextFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter *param) const {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  DLOG(INFO) << "Serializing " << layers_.size() << " layers";
  for (size_t i = 0; i < layers_.size(); ++i) {
    LayerParameter *layer_param = param->add_layer();
    layers_[i]->ToProto(layer_param);
  }
}

template <typename Dtype>
void Net<Dtype>::ShareWeights() {
  for (size_t i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] < 0) {
      continue;
    }
    params_[i]->ShareData(*params_[param_owners_[i]]);
  }
}

template <typename Dtype>
bool Net<Dtype>::has_blob(const string &blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Blob<Dtype> > Net<Dtype>::blob_by_name(
    const string &blob_name) const {
  shared_ptr<Blob<Dtype> > blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  } else {
    blob_ptr.reset((Blob<Dtype> *)(NULL));
    LOG(WARNING) << "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_layer(const string &layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Layer<Dtype> > Net<Dtype>::layer_by_name(
    const string &layer_name) const {
  shared_ptr<Layer<Dtype> > layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  } else {
    layer_ptr.reset((Layer<Dtype> *)(NULL));
    LOG(WARNING) << "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}

INSTANTIATE_CLASS(Net);

}  // namespace latte
