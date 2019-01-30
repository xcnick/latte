#ifndef LATTE_UTIL_INSERT_SPLITS_H_
#define LATTE_UTIL_INSERT_SPLITS_H_

#include "latte/proto/latte.pb.h"
#include "latte/common.h"

namespace latte {

// 对于底层一个输出blob对应多个上层的情况，则要在加入分裂层，形成新的网络。
// 这么做的主要原因是多个层反传给该blob的梯度需要累加。
void InsertSplits(const NetParameter &param, NetParameter *param_split);

void ConfigureSplitLayer(const string &layer_name, const string &blob_name,
                         const int blob_idx, const int split_count,
                         const float loss_weight,
                         LayerParameter *split_layer_param);

string SplitLayerName(const string &layer_name, const string &blob_name,
                      const int blob_idx);

string SplitBlobName(const string &layer_name, const string &blob_name,
                     const int blob_idx, const int split_idx);
}  // namespace latte

#endif  // LATTE_UTIL_INSERT_SPLITS_H_