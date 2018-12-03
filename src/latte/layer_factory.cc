#include "latte/layer_factory.h"
#include "latte/proto/latte.pb.h"

#include "latte/layers/sigmoid_layer.h"

#ifdef USE_CUDNN
#include "latte/layers/cudnn_sigmoid_layer.h"
#endif

namespace latte {

template <typename Dtype>
shared_ptr<Layer<Dtype>> GetSigmoidLayer(const LayerParameter &param) {
  SigmoidParameter_Engine engine = param.sigmoid_param().engine();
  if (engine == SigmoidParameter_Engine_DEFAULT) {
    engine = SigmoidParameter_Engine_LATTE;
#ifdef USE_CUDNN
    engine = SigmoidParameter_Engine_CUDNN;
#endif
  }
  if (engine == SigmoidParameter_Engine_LATTE) {
    return shared_ptr<Layer<Dtype>>(new SigmoidLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == SigmoidParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype>>(new CuDNNSigmoidLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;
  }
}

REGISTER_LAYER_CREATOR(Sigmoid, GetSigmoidLayer);

}  // namespace latte
