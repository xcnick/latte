#include "latte/common.h"
#include "latte/filler.h"
#include "latte/layers/softmax_layer.h"
#include "latte/test/test_latte_main.h"

#ifdef USE_CUDNN
#include "latte/layers/cudnn_softmax_layer.h"
#endif

namespace latte {

template <typename TypeParam>
class SoftmaxLayerTest : public MultiDeviceTest<TypeParam> {
  using Dtype = typename TypeParam::Dtype;

 protected:
  SoftmaxLayerTest()
      : blob_bottom_(new Blob<Dtype>({2, 10, 2, 3})),
        blob_top_(new Blob<Dtype>()) {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~SoftmaxLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype> *const blob_bottom_;
  Blob<Dtype> *const blob_top_;
  vector<Blob<Dtype> *> blob_bottom_vec_;
  vector<Blob<Dtype> *> blob_top_vec_;
};

TYPED_TEST_SUITE(SoftmaxLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxLayerTest, TestForward) {
  using Dtype = typename TypeParam::Dtype;
  LayerParameter layer_param;
  SoftmaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->shape(0); ++i) {
    for (int k = 0; k < this->blob_bottom_->shape(2); ++k) {
      for (int l = 0; l < this->blob_bottom_->shape(3); ++l) {
        Dtype sum = 0;
        for (int j = 0; j < this->blob_bottom_->shape(1); ++j) {
          sum += this->blob_top_->data_at({i, j, k, l});
        }
        EXPECT_GE(sum, 0.999);
        EXPECT_LE(sum, 1.001);

        Dtype scale = 0;
        for (int j = 0; j < this->blob_bottom_->shape(1); ++j) {
          scale += exp(this->blob_bottom_->data_at({i, j, k, l}));
        }
        for (int j = 0; j < this->blob_bottom_->shape(1); ++j) {
          EXPECT_GE(this->blob_top_->data_at({i, j, k, l}) + 1e-4,
                    exp(this->blob_bottom_->data_at({i, j, k, l})) / scale)
              << "debug: " << i << " " << j;
          EXPECT_LE(this->blob_top_->data_at({i, j, k, l}) - 1e-4,
                    exp(this->blob_bottom_->data_at({i, j, k, l})) / scale)
              << "debug: " << i << " " << j;
        }
      }
    }
  }
}

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNSoftmaxLayerTest : public GPUDeviceTest<Dtype> {
 protected:
  CuDNNSoftmaxLayerTest()
      : blob_bottom_(new Blob<Dtype>({2, 10, 2, 3})),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~CuDNNSoftmaxLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype> *const blob_bottom_;
  Blob<Dtype> *const blob_top_;
  vector<Blob<Dtype> *> blob_bottom_vec_;
  vector<Blob<Dtype> *> blob_top_vec_;
};

TYPED_TEST_SUITE(CuDNNSoftmaxLayerTest, TestDtypes);

TYPED_TEST(CuDNNSoftmaxLayerTest, TestForwardCuDNN) {
  LayerParameter layer_param;
  CuDNNSoftmaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test sum
  for (int i = 0; i < this->blob_bottom_->shape(0); ++i) {
    for (int k = 0; k < this->blob_bottom_->shape(2); ++k) {
      for (int l = 0; l < this->blob_bottom_->shape(3); ++l) {
        TypeParam sum = 0;
        for (int j = 0; j < this->blob_top_->shape(1); ++j) {
          sum += this->blob_top_->data_at({i, j, k, l});
        }
        EXPECT_GE(sum, 0.999);
        EXPECT_LE(sum, 1.001);
        // Test exact values
        TypeParam scale = 0;
        for (int j = 0; j < this->blob_bottom_->shape(1); ++j) {
          scale += exp(this->blob_bottom_->data_at({i, j, k, l}));
        }
        for (int j = 0; j < this->blob_bottom_->shape(1); ++j) {
          EXPECT_GE(this->blob_top_->data_at({i, j, k, l}) + 1e-4,
                    exp(this->blob_bottom_->data_at({i, j, k, l})) / scale)
              << "debug: " << i << " " << j;
          EXPECT_LE(this->blob_top_->data_at({i, j, k, l}) - 1e-4,
                    exp(this->blob_bottom_->data_at({i, j, k, l})) / scale)
              << "debug: " << i << " " << j;
        }
      }
    }
  }
}

#endif

}  // namespace latte