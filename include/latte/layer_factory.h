/**
 * LayerRegistry<Dtype>::CreateLayer(param);
 *
 * 1:
 *  template <typename Dtype>
 *  class MyAwesomeLayer : public Layer<Dtype> {
 *    // your implementations
 *  };
 *  REGISTER_LAYER_CLASS(MyAwesome);
 *
 * 2:
 *  template <typename Dtype>
 *  Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *    // your implementation
 *  }
 *  REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 */

#ifndef LATTE_LAYER_FACTORY_H_
#define LATTE_LAYER_FACTORY_H_

#include "latte/common.h"
#include "latte/layer.h"
#include "latte/proto/latte.pb.h"

namespace latte {

template <typename Dtype>
class LayerRegistry {
 public:
  using Creator = shared_ptr<Layer<Dtype>> (*)(const LayerParameter &);
  using CreatorRegistry = std::map<string, Creator>;

  static CreatorRegistry &Registry() {
    static CreatorRegistry instance;
    return instance;
  }

  // Adds a creator.
  static void AddCreator(const string &type, Creator creator) {
    CreatorRegistry &registry = Registry();
    CHECK_EQ(registry.count(type), 0)
        << "Layer type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a layer using a LayerParameter
  static shared_ptr<Layer<Dtype>> CreateLayer(const LayerParameter &param) {
    if (Latte::root_solver()) {
      LOG(INFO) << "Creating layer " << param.name();
    }
    const string &type = param.type();
    CreatorRegistry &registry = Registry();
    CHECK_EQ(registry.count(type), 1)
        << "Unknown layer type: " << type
        << " (known types: " << LayerTypeListString() << ")";
    return registry[type](param);
  }

  static vector<string> LayerTypeList() {
    CreatorRegistry &registry = Registry();
    vector<string> layer_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      layer_types.push_back(iter->first);
    }
    return layer_types;
  }

 private:
  LayerRegistry() = delete;

  static string LayerTypeListString() {
    vector<string> layer_types = LayerTypeList();
    string layer_types_str;
    for (vector<string>::iterator iter = layer_types.begin();
         iter != layer_types.end(); ++iter) {
      if (iter != layer_types.begin()) {
        layer_types_str += ", ";
      }
      layer_types_str += *iter;
    }
    return layer_types_str;
  }
};

template <typename Dtype>
class LayerRegisterer {
 public:
  LayerRegisterer(const string &type,
                  shared_ptr<Layer<Dtype>> (*creator)(const LayerParameter &)) {
    LayerRegistry<Dtype>::AddCreator(type, creator);
  }
};

#define REGISTER_LAYER_CREATOR(type, creator)                               \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>); \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)

#define REGISTER_LAYER_CLASS(type)                                  \
  template <typename Dtype>                                         \
  shared_ptr<Layer<Dtype>> Creator_##type##Layer(                   \
      const LayerParameter &param) {                                \
    return shared_ptr<Layer<Dtype>>(new type##Layer<Dtype>(param)); \
  }                                                                 \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

}  // namespace latte

#endif  // LATTE_LAYER_FACTORY_H_