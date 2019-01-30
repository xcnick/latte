#include "latte/filler.h"
#include "latte/test/test_latte_main.h"

namespace latte {

template <typename Dtype>
class ConstantFillerTest : public ::testing::Test {
 protected:
  ConstantFillerTest() : blob_(new Blob<Dtype>()), filler_param_() {
    filler_param_.set_value(10.);
    filler_.reset(new ConstantFiller<Dtype>(filler_param_));
  }

  virtual void test_params(const vector<int> &shape) {
    EXPECT_TRUE(blob_);
    blob_->Reshape(shape);
    filler_->Fill(blob_);
    const int count = blob_->count();
    const Dtype *data = blob_->cpu_data();
    for (int i = 0; i < count; ++i) {
      EXPECT_EQ(data[i], filler_param_.value());
    }
  }

  virtual ~ConstantFillerTest() { delete blob_; }

  Blob<Dtype> *const blob_;
  FillerParameter filler_param_;
  shared_ptr<ConstantFiller<Dtype>> filler_;
};

TYPED_TEST_CASE(ConstantFillerTest, TestDtypes);

TYPED_TEST(ConstantFillerTest, TestFill) {
  vector<int> blob_shape;
  blob_shape.push_back(2);
  blob_shape.push_back(3);
  blob_shape.push_back(4);
  blob_shape.push_back(5);
  this->test_params(blob_shape);
}

TYPED_TEST(ConstantFillerTest, TestFill1D) {
  vector<int> blob_shape(1, 15);
  this->test_params(blob_shape);
}

TYPED_TEST(ConstantFillerTest, TestFill2D) {
  vector<int> blob_shape;
  blob_shape.push_back(8);
  blob_shape.push_back(3);
  this->test_params(blob_shape);
}

TYPED_TEST(ConstantFillerTest, TestFill5D) {
  vector<int> blob_shape;
  blob_shape.push_back(2);
  blob_shape.push_back(3);
  blob_shape.push_back(4);
  blob_shape.push_back(5);
  blob_shape.push_back(2);
  this->test_params(blob_shape);
}

}  // namespace latte