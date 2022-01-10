#ifndef LATTE_DUMMY_DATA_LAYER_H_
#define LATTE_DUMMY_DATA_LAYER_H_

#include "latte/blob.h"
#include "latte/filler.h"
#include "latte/layer.h"
#include "latte/proto/latte.pb.h"

namespace latte {

template <typename Dtype>
class DummyDataLayer : public Layer<Dtype> {
 public:
  explicit DummyDataLayer(const LayerParameter &param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top) {}

  virtual inline const char *type() const { return "DummyData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);

  vector<shared_ptr<Filler<Dtype> > > fillers_;
  vector<bool> refill_;
};

}  // namespace latte

#endif  // LATTE_DUMMY_DATA_LAYER_H_