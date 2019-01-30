#include "latte/common.h"
#include "latte/filler.h"
#include "latte/layers/dummy_data_layer.h"
#include "latte/test/test_latte_main.h"

namespace latte {

template <typename Dtype>
class DummyDataLayerTest : public CPUDeviceTest<Dtype> {
 protected:
  DummyDataLayerTest()
      : blob_top_a_(new Blob<Dtype>()),
        blob_top_b_(new Blob<Dtype>()),
        blob_top_c_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    blob_bottom_vec_.clear();
    blob_top_vec_.clear();
    blob_top_vec_.push_back(blob_top_a_);
    blob_top_vec_.push_back(blob_top_b_);
    blob_top_vec_.push_back(blob_top_c_);
  }

  virtual ~DummyDataLayerTest() {
    delete blob_top_a_;
    delete blob_top_b_;
    delete blob_top_c_;
  }

  Blob<Dtype> *const blob_top_a_;
  Blob<Dtype> *const blob_top_b_;
  Blob<Dtype> *const blob_top_c_;
  vector<Blob<Dtype> *> blob_bottom_vec_;
  vector<Blob<Dtype> *> blob_top_vec_;
};

TYPED_TEST_CASE(DummyDataLayerTest, TestDtypes);

TYPED_TEST(DummyDataLayerTest, TestOneTopConstant) {
  LayerParameter param;
  DummyDataParameter *dummy_data_param = param.mutable_dummy_data_param();
  dummy_data_param->add_shape();
  dummy_data_param->mutable_shape(0)->add_dim(5);
  dummy_data_param->mutable_shape(0)->add_dim(3);
  dummy_data_param->mutable_shape(0)->add_dim(2);
  dummy_data_param->mutable_shape(0)->add_dim(4);
  this->blob_top_vec_.resize(1);
  DummyDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_a_->shape(0), 5);
  EXPECT_EQ(this->blob_top_a_->shape(1), 3);
  EXPECT_EQ(this->blob_top_a_->shape(2), 2);
  EXPECT_EQ(this->blob_top_a_->shape(3), 4);
  EXPECT_EQ(this->blob_top_b_->count(), 0);
  EXPECT_EQ(this->blob_top_c_->count(), 0);
  for (int i = 0; i < this->blob_top_vec_.size(); ++i) {
    for (int j = 0; j < this->blob_top_vec_[i]->count(); ++j) {
      EXPECT_EQ(0, this->blob_top_vec_[i]->cpu_data()[j]);
    }
  }
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_vec_.size(); ++i) {
    for (int j = 0; j < this->blob_top_vec_[i]->count(); ++j) {
      EXPECT_EQ(0, this->blob_top_vec_[i]->cpu_data()[j]);
    }
  }
}

TYPED_TEST(DummyDataLayerTest, TestTwoTopConstant) {
  LayerParameter param;
  DummyDataParameter *dummy_data_param = param.mutable_dummy_data_param();
  dummy_data_param->add_shape();
  dummy_data_param->mutable_shape(0)->add_dim(5);
  dummy_data_param->mutable_shape(0)->add_dim(3);
  dummy_data_param->mutable_shape(0)->add_dim(2);
  dummy_data_param->mutable_shape(0)->add_dim(4);
  dummy_data_param->add_shape();
  dummy_data_param->mutable_shape(1)->add_dim(5);
  dummy_data_param->mutable_shape(1)->add_dim(3);
  dummy_data_param->mutable_shape(1)->add_dim(1);
  dummy_data_param->mutable_shape(1)->add_dim(4);
  FillerParameter *data_filler_param = dummy_data_param->add_data_filler();
  data_filler_param->set_type("constant");
  data_filler_param->set_value(7);
  this->blob_top_vec_.resize(2);
  DummyDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_a_->shape(0), 5);
  EXPECT_EQ(this->blob_top_a_->shape(1), 3);
  EXPECT_EQ(this->blob_top_a_->shape(2), 2);
  EXPECT_EQ(this->blob_top_a_->shape(3), 4);
  EXPECT_EQ(this->blob_top_b_->shape(0), 5);
  EXPECT_EQ(this->blob_top_b_->shape(1), 3);
  EXPECT_EQ(this->blob_top_b_->shape(2), 1);
  EXPECT_EQ(this->blob_top_b_->shape(3), 4);
  EXPECT_EQ(this->blob_top_c_->count(), 0);
  for (int i = 0; i < this->blob_top_vec_.size(); ++i) {
    for (int j = 0; j < this->blob_top_vec_[i]->count(); ++j) {
      EXPECT_EQ(7, this->blob_top_vec_[i]->cpu_data()[j]);
    }
  }
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_vec_.size(); ++i) {
    for (int j = 0; j < this->blob_top_vec_[i]->count(); ++j) {
      EXPECT_EQ(7, this->blob_top_vec_[i]->cpu_data()[j]);
    }
  }
}

}  // namespace latte