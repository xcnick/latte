#include "latte/common.h"
#include "latte/filler.h"
#include "latte/layers/inner_product_layer.h"
#include "latte/test/test_latte_main.h"

namespace latte {

template <typename TypeParam>
class InnerProductLayerTest : public MultiDeviceTest<TypeParam> {
  using Dtype = typename TypeParam::Dtype;

 protected:
  InnerProductLayerTest()
      : blob_bottom_(new Blob<Dtype>({2, 3, 4, 5})),
        blob_bottom_nobatch_(new Blob<Dtype>({1, 2, 3, 4})),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~InnerProductLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_nobatch_;
    delete blob_top_;
  }

  Blob<Dtype> *const blob_bottom_;
  Blob<Dtype> *const blob_bottom_nobatch_;
  Blob<Dtype> *const blob_top_;
  vector<Blob<Dtype> *> blob_bottom_vec_;
  vector<Blob<Dtype> *> blob_top_vec_;
};

TYPED_TEST_SUITE(InnerProductLayerTest, TestDtypesAndDevices);

TYPED_TEST(InnerProductLayerTest, TestSetUp) {
  using Dtype = typename TypeParam::Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter *inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  shared_ptr<InnerProductLayer<Dtype>> layer(
      new InnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num_axes(), 2);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 10);
}

TYPED_TEST(InnerProductLayerTest, TestSetUpTransposeFalse) {
  using Dtype = typename TypeParam::Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter *inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_transpose()->set_value(false);
  shared_ptr<InnerProductLayer<Dtype>> layer(
      new InnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num_axes(), 2);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 10);
  EXPECT_EQ(layer->blobs()[0]->num_axes(), 2);
  EXPECT_EQ(layer->blobs()[0]->shape(0), 10);
  EXPECT_EQ(layer->blobs()[0]->shape(1), 60);
}

TYPED_TEST(InnerProductLayerTest, TestSetUpTransposeTrue) {
  using Dtype = typename TypeParam::Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter *inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_transpose()->set_value(true);
  shared_ptr<InnerProductLayer<Dtype>> layer(
      new InnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num_axes(), 2);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 10);
  EXPECT_EQ(layer->blobs()[0]->num_axes(), 2);
  EXPECT_EQ(layer->blobs()[0]->shape(0), 60);
  EXPECT_EQ(layer->blobs()[0]->shape(1), 10);
}

TYPED_TEST(InnerProductLayerTest, TestForward) {
  using Dtype = typename TypeParam::Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  if (Latte::mode() == Latte::CPU || sizeof(Dtype) == 4) {
    LayerParameter layer_param;
    InnerProductParameter *inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);
    shared_ptr<InnerProductLayer<Dtype>> layer(
        new InnerProductLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_->num_axes(), 2);
    EXPECT_EQ(this->blob_top_->shape(0), 2);
    EXPECT_EQ(this->blob_top_->shape(1), 10);
    const Dtype *data = this->blob_top_->cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_GE(data[i], 1);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(InnerProductLayerTest, TestForwardTranspose) {
  using Dtype = typename TypeParam::Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  if (Latte::mode() == Latte::CPU || sizeof(Dtype) == 4) {
    LayerParameter layer_param;
    InnerProductParameter *inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);
    inner_product_param->mutable_transpose()->set_value(false);
    shared_ptr<InnerProductLayer<Dtype>> layer(
        new InnerProductLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_->num_axes(), 2);
    EXPECT_EQ(this->blob_top_->shape(0), 2);
    EXPECT_EQ(this->blob_top_->shape(1), 10);
    const int count = this->blob_top_->count();
    Blob<Dtype> *const top = new Blob<Dtype>();
    top->ReshapeLike(*this->blob_top_);
    latte_copy(count, this->blob_top_->cpu_data(), top->mutable_cpu_data());
    this->blob_top_vec_.clear();
    this->blob_top_vec_.push_back(new Blob<Dtype>());
    inner_product_param->mutable_transpose()->set_value(true);
    shared_ptr<InnerProductLayer<Dtype>> ip_t(
        new InnerProductLayer<Dtype>(layer_param));
    ip_t->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const int count_w = layer->blobs()[0]->count();
    EXPECT_EQ(ip_t->blobs()[0]->count(), count_w);
    // 将weight转置并从layer拷贝到ip_t中
    const Dtype *w = layer->blobs()[0]->cpu_data();
    Dtype *w_t = ip_t->blobs()[0]->mutable_cpu_data();
    const int width = layer->blobs()[0]->shape(1);
    const int width_t = ip_t->blobs()[0]->shape(1);
    for (int i = 0; i < count_w; ++i) {
      int r = i / width;
      int c = i % width;
      w_t[c * width_t + r] = w[r * width + c];
    }
    // 将bias值从layer拷贝到ip_t中
    ASSERT_EQ(layer->blobs()[1]->count(), ip_t->blobs()[1]->count());
    latte_copy(layer->blobs()[1]->count(), layer->blobs()[1]->cpu_data(),
               ip_t->blobs()[1]->mutable_cpu_data());
    ip_t->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_->count(), count)
        << "Invalid count for top blob for IP with transpose.";
    Blob<Dtype> *const top_t = new Blob<Dtype>();
    top_t->ReshapeLike(*this->blob_top_vec_[0]);
    latte_copy(count, this->blob_top_vec_[0]->cpu_data(),
               top_t->mutable_cpu_data());
    const Dtype *data = top->cpu_data();
    const Dtype *data_t = top_t->cpu_data();
    for (int i = 0; i < count; ++i) {
      EXPECT_FLOAT_EQ(data[i], data_t[i]);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(InnerProductLayerTest, TestForwardNoBatch) {
  using Dtype = typename TypeParam::Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_nobatch_);
  if (Latte::mode() == Latte::CPU || sizeof(Dtype) == 4) {
    LayerParameter layer_param;
    InnerProductParameter *inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(10);
    inner_product_param->mutable_weight_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);
    shared_ptr<InnerProductLayer<Dtype>> layer(
        new InnerProductLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_EQ(this->blob_top_->num_axes(), 2);
    EXPECT_EQ(this->blob_top_->shape(0), 1);
    EXPECT_EQ(this->blob_top_->shape(1), 10);
    const Dtype *data = this->blob_top_->cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_GE(data[i], 1.);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace latte