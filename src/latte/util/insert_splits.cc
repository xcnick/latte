#include "latte/util/insert_splits.h"

namespace latte {

void InsertSplits(const NetParameter &param, NetParameter *param_split) {
  param_split->CopyFrom(param);
  param_split->clear_layer();
  // blob_name_to_last_top_idx[“conv1”]=(1,0)
  // 这个例子相当于说”conv1” 这层是第1层的第0个top
  map<string, pair<int, int>> blob_name_to_last_top_idx;
  // bottom_idx_to_source_top_idx[bottom_idx=(2,0)] = (1,0);
  // 相当于说：第2层，第0个bottom，对应着第1层，第0个top
  map<pair<int, int>, pair<int, int>> bottom_idx_to_source_top_idx;
  // top_idx_to_bottom_count[(1,0)]=2
  // 表示第1个layer的第0个top有2个top(blob)，要分叉，程序会对此建立新的分叉层
  map<pair<int, int>, int> top_idx_to_bottom_count;
  map<pair<int, int>, int> top_idx_to_bottom_split_idx;
  // 记录各层的名称，如 [0x00000000] "input"
  map<int, string> layer_idx_to_layer_name;
  // 遍历整个网络，记录每一个Layer的top的使用情况，记录结构放在
  // top_idx_to_bottom_count中。
  for (int i = 0; i < param.layer_size(); ++i) {
    const LayerParameter &layer_param = param.layer(i);
    layer_idx_to_layer_name[i] = layer_param.name();
    for (int j = 0; j < layer_param.bottom_size(); ++j) {
      const string &blob_name = layer_param.bottom(j);
      if (blob_name_to_last_top_idx.find(blob_name) ==
          blob_name_to_last_top_idx.end()) {
        LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
                   << layer_param.name() << "', bottom index " << j << ")";
      }
      const pair<int, int> &bottom_idx = make_pair(i, j);
      const pair<int, int> &top_idx = blob_name_to_last_top_idx[blob_name];
      bottom_idx_to_source_top_idx[bottom_idx] = top_idx;
      ++top_idx_to_bottom_count[top_idx];
    }
    for (int j = 0; j < layer_param.top_size(); ++j) {
      const string &blob_name = layer_param.top(j);
      blob_name_to_last_top_idx[blob_name] = make_pair(i, j);
    }
  }
  for (int i = 0; i < param.layer_size(); ++i) {
    LayerParameter *layer_param = param_split->add_layer();
    layer_param->CopyFrom(param.layer(i));
    // Replace any shared bottom blobs with split layer outputs.
    for (int j = 0; j < layer_param->bottom_size(); ++j) {
      const pair<int, int> &top_idx =
          bottom_idx_to_source_top_idx[make_pair(i, j)];
      const int split_count = top_idx_to_bottom_count[top_idx];
      if (split_count > 1) {
        const string &layer_name = layer_idx_to_layer_name[top_idx.first];
        const string &blob_name = layer_param->bottom(j);
        layer_param->set_bottom(
            j, SplitBlobName(layer_name, blob_name, top_idx.second,
                             top_idx_to_bottom_split_idx[top_idx]++));
      }
    }
    // Create split layer for any top blobs used by other layer as bottom
    // blobs more than once.
    for (int j = 0; j < layer_param->top_size(); ++j) {
      const pair<int, int> &top_idx = make_pair(i, j);
      const int split_count = top_idx_to_bottom_count[top_idx];
      if (split_count > 1) {
        const string &layer_name = layer_idx_to_layer_name[i];
        const string &blob_name = layer_param->top(j);
        LayerParameter *split_layer_param = param_split->add_layer();
        ConfigureSplitLayer(layer_name, blob_name, j, split_count,
                            split_layer_param);
      }
    }
  }
}

void ConfigureSplitLayer(const string &layer_name, const string &blob_name,
                         const int blob_idx, const int split_count,
                         LayerParameter *split_layer_param) {
  split_layer_param->Clear();
  split_layer_param->add_bottom(blob_name);
  split_layer_param->set_name(SplitLayerName(layer_name, blob_name, blob_idx));
  split_layer_param->set_type("Split");
  for (int k = 0; k < split_count; ++k) {
    split_layer_param->add_top(
        SplitBlobName(layer_name, blob_name, blob_idx, k));
  }
}

string SplitLayerName(const string &layer_name, const string &blob_name,
                      const int blob_idx) {
  ostringstream split_layer_name;
  split_layer_name << blob_name << "_" << layer_name << "_" << blob_idx
                   << "_split";
  return split_layer_name.str();
}

string SplitBlobName(const string &layer_name, const string &blob_name,
                     const int blob_idx, const int split_idx) {
  ostringstream split_blob_name;
  split_blob_name << blob_name << "_" << layer_name << "_" << blob_idx
                  << "_split_" << split_idx;
  return split_blob_name.str();
}

}  // namespace latte