#include "latte/common.h"
#include "latte/layer.h"
#include "latte/layer_factory.h"

#include "latte/test/test_latte_main.h"

namespace latte {

template <typename TypeParam>
class LayerFactoryTest : public MultiDeviceTest<TypeParam> {};

TYPED_TEST_SUITE(LayerFactoryTest, TestDtypesAndDevices);

TYPED_TEST(LayerFactoryTest, TestCreateLayer) {
  using Dtype = typename TypeParam::Dtype;
  auto &registry = LayerRegistry<Dtype>::Registry();
  shared_ptr<Layer<Dtype>> layer;
  for (auto iter = registry.begin(); iter != registry.end(); ++iter) {
    LayerParameter layer_param;
    layer_param.set_type(iter->first);
    layer = LayerRegistry<Dtype>::CreateLayer(layer_param);
    EXPECT_EQ(iter->first, layer->type());
  }
}
}  // namespace latte