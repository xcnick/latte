#include "latte/blob.h"
#include "latte/common.h"
#include "latte/filler.h"

#include "latte/test/test_latte_main.h"

namespace latte {

template <typename Dtype>
class BlobSimpleTest : public ::testing::Test {
 protected:
  BlobSimpleTest()
      : blob_(new Blob<Dtype>()),
        blob_preshaped_(new Blob<Dtype>({2, 3, 4, 5})) {}

  virtual ~BlobSimpleTest() {
    delete blob_;
    delete blob_preshaped_;
  }

  Blob<Dtype> *const blob_;
  Blob<Dtype> *const blob_preshaped_;
};

TYPED_TEST_CASE(BlobSimpleTest, TestDtypes);

TYPED_TEST(BlobSimpleTest, TestInitialization) {
  EXPECT_TRUE(this->blob_);
  EXPECT_TRUE(this->blob_preshaped_);
  EXPECT_EQ(this->blob_preshaped_->shape(0), 2);
  EXPECT_EQ(this->blob_preshaped_->shape(1), 3);
  EXPECT_EQ(this->blob_preshaped_->shape(2), 4);
  EXPECT_EQ(this->blob_preshaped_->shape(3), 5);
  EXPECT_EQ(this->blob_preshaped_->count(), 120);
  EXPECT_EQ(this->blob_preshaped_->num_axes(), 4);
  EXPECT_EQ(this->blob_->num_axes(), 0);
  EXPECT_EQ(this->blob_->count(), 0);
}

TYPED_TEST(BlobSimpleTest, TestPointerCPUGPU) {
  EXPECT_TRUE(this->blob_preshaped_->gpu_data());
  EXPECT_TRUE(this->blob_preshaped_->cpu_data());
  EXPECT_TRUE(this->blob_preshaped_->mutable_gpu_data());
  EXPECT_TRUE(this->blob_preshaped_->mutable_cpu_data());
}

TYPED_TEST(BlobSimpleTest, TestReshape) {
  this->blob_->Reshape({2, 3, 4, 5});
  EXPECT_EQ(this->blob_->shape(0), 2);
  EXPECT_EQ(this->blob_->shape(1), 3);
  EXPECT_EQ(this->blob_->shape(2), 4);
  EXPECT_EQ(this->blob_->shape(3), 5);
  EXPECT_EQ(this->blob_->count(), 120);
}

TYPED_TEST(BlobSimpleTest, TestReshapeZero) {
  vector<int> shape(2);
  shape[0] = 0;
  shape[1] = 5;
  this->blob_->Reshape(shape);
  EXPECT_EQ(this->blob_->count(), 0);
}

template <typename TypeParam>
class BlobMathTest : public MultiDeviceTest<TypeParam> {
  using Dtype = typename TypeParam::Dtype;

 protected:
  BlobMathTest() : blob_(new Blob<Dtype>({2, 3, 4, 5})), epsilon_(1e-6) {}
  virtual ~BlobMathTest() { delete blob_; }
  Blob<Dtype> *const blob_;
  Dtype epsilon_;
};

TYPED_TEST_CASE(BlobMathTest, TestDtypesAndDevices);

TYPED_TEST(BlobMathTest, TestSumOfSquares) {
  using Dtype = typename TypeParam::Dtype;

  EXPECT_EQ(0, this->blob_->sumsq_data());
  EXPECT_EQ(0, this->blob_->sumsq_diff());
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_);
  Dtype expected_sumsq = 0;
  const Dtype *data = this->blob_->cpu_data();
  for (int i = 0; i < this->blob_->count(); ++i) {
    expected_sumsq += data[i] * data[i];
  }
  switch (TypeParam::device) {
    case Latte::CPU:
      this->blob_->mutable_cpu_data();
      break;
    case Latte::GPU:
      this->blob_->mutable_gpu_data();
      break;
    default:
      LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  EXPECT_NEAR(expected_sumsq, this->blob_->sumsq_data(),
              this->epsilon_ * expected_sumsq);
  EXPECT_EQ(0, this->blob_->sumsq_diff());

  // Check sumsq
  const Dtype kDiffScaleFactor = 7;
  latte_cpu_scale(this->blob_->count(), kDiffScaleFactor, data,
                  this->blob_->mutable_cpu_diff());
  switch (TypeParam::device) {
    case Latte::CPU:
      this->blob_->mutable_cpu_diff();
      break;
    case Latte::GPU:
      this->blob_->mutable_gpu_diff();
      break;
    default:
      LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  EXPECT_NEAR(expected_sumsq, this->blob_->sumsq_data(),
              this->epsilon_ * expected_sumsq);
  const Dtype expected_sumsq_diff =
      expected_sumsq * kDiffScaleFactor * kDiffScaleFactor;
  EXPECT_NEAR(expected_sumsq_diff, this->blob_->sumsq_diff(),
              this->epsilon_ * expected_sumsq_diff);
}

TYPED_TEST(BlobMathTest, TestAsum) {
  using Dtype = typename TypeParam::Dtype;

  EXPECT_EQ(0, this->blob_->asum_data());
  EXPECT_EQ(0, this->blob_->asum_diff());
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_);
  Dtype expected_asum = 0;
  const Dtype *data = this->blob_->cpu_data();
  for (int i = 0; i < this->blob_->count(); ++i) {
    expected_asum += std::fabs(data[i]);
  }
  switch (TypeParam::device) {
    case Latte::CPU:
      this->blob_->mutable_cpu_data();
      break;
    case Latte::GPU:
      this->blob_->mutable_gpu_data();
      break;
    default:
      LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  EXPECT_NEAR(expected_asum, this->blob_->asum_data(),
              this->epsilon_ * expected_asum);
  EXPECT_EQ(0, this->blob_->asum_diff());

  // check diff
  const Dtype kDiffScaleFactor = 7;
  latte_cpu_scale(this->blob_->count(), kDiffScaleFactor, data,
                  this->blob_->mutable_cpu_diff());
  switch (TypeParam::device) {
    case Latte::CPU:
      this->blob_->mutable_cpu_diff();
      break;
    case Latte::GPU:
      this->blob_->mutable_gpu_diff();
      break;
    default:
      LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  EXPECT_NEAR(expected_asum, this->blob_->asum_data(),
              this->epsilon_ * expected_asum);
  const Dtype expected_diff_asum = expected_asum * kDiffScaleFactor;
  EXPECT_NEAR(expected_diff_asum, this->blob_->asum_diff(),
              this->epsilon_ * expected_diff_asum);
}

TYPED_TEST(BlobMathTest, TestScale) {
  using Dtype = typename TypeParam::Dtype;

  EXPECT_EQ(0, this->blob_->asum_data());
  EXPECT_EQ(0, this->blob_->asum_diff());
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_);
  const Dtype asum_before_scale = this->blob_->asum_data();
  switch (TypeParam::device) {
    case Latte::CPU:
      this->blob_->mutable_cpu_data();
      break;
    case Latte::GPU:
      this->blob_->mutable_gpu_data();
      break;
    default:
      LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  const Dtype kDataScaleFactor = 3;
  this->blob_->scale_data(kDataScaleFactor);
  EXPECT_NEAR(asum_before_scale * kDataScaleFactor, this->blob_->asum_data(),
              this->epsilon_ * asum_before_scale * kDataScaleFactor);
  EXPECT_EQ(0, this->blob_->asum_diff());

  // check diff
  const Dtype kDataToDiffScaleFactor = 7;
  const Dtype *data = this->blob_->cpu_data();
  latte_cpu_scale(this->blob_->count(), kDataToDiffScaleFactor, data,
                  this->blob_->mutable_cpu_diff());
  const Dtype expected_asum_before_scale = asum_before_scale * kDataScaleFactor;
  EXPECT_NEAR(expected_asum_before_scale, this->blob_->asum_data(),
              this->epsilon_ * expected_asum_before_scale);
  const Dtype expected_diff_asum_before_scale =
      asum_before_scale * kDataScaleFactor * kDataToDiffScaleFactor;
  EXPECT_NEAR(expected_diff_asum_before_scale, this->blob_->asum_diff(),
              this->epsilon_ * expected_diff_asum_before_scale);
  switch (TypeParam::device) {
    case Latte::CPU:
      this->blob_->mutable_cpu_diff();
      break;
    case Latte::GPU:
      this->blob_->mutable_gpu_diff();
      break;
    default:
      LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  const Dtype kDiffScaleFactor = 3;
  this->blob_->scale_diff(kDiffScaleFactor);
  EXPECT_NEAR(asum_before_scale * kDiffScaleFactor, this->blob_->asum_data(),
              this->epsilon_ * asum_before_scale * kDiffScaleFactor);
  const Dtype expected_diff_asum =
      expected_diff_asum_before_scale * kDiffScaleFactor;
  EXPECT_NEAR(expected_diff_asum, this->blob_->asum_diff(),
              this->epsilon_ * expected_diff_asum);
}

}  // namespace latte