#include "latte/common.h"
#include "latte/filler.h"
#include "latte/layers/euclidean_loss_layer.h"
#include "latte/test/test_latte_main.h"

namespace latte {

template <typename TypeParam>
class EuclideanLossLayerTest : public MultiDeviceTest<TypeParam> {
  using Dtype = typename TypeParam::Dtype;

 protected:
  EuclideanLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>({10, 5, 1, 1})),
        blob_bottom_label_(new Blob<Dtype>({10, 5, 1, 1})),
        blob_top_loss_(new Blob<Dtype>()) {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }

  virtual ~EuclideanLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }

  void TestForward() {
    LayerParameter layer_param;
    EuclideanLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    EuclideanLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);

    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
  }

  Blob<Dtype> *const blob_bottom_data_;
  Blob<Dtype> *const blob_bottom_label_;
  Blob<Dtype> *const blob_top_loss_;
  vector<Blob<Dtype> *> blob_bottom_vec_;
  vector<Blob<Dtype> *> blob_top_vec_;
};

TYPED_TEST_SUITE(EuclideanLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(EuclideanLossLayerTest, TestForward) { this->TestForward(); }

TYPED_TEST(EuclideanLossLayerTest, TestGradient) {

}

}  // namespace latte