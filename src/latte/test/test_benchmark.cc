#include <thread>

#include "latte/common.h"
#include "latte/util/benchmark.h"

#include "latte/test/test_latte_main.h"

namespace latte {

const float kMillisecondsThreshold = 30;

template <typename TypeParam>
class BenchmarkTest : public MultiDeviceTest<TypeParam> {};

TYPED_TEST_CASE(BenchmarkTest, TestDtypesAndDevices);

TYPED_TEST(BenchmarkTest, TestTimerConstructor) {
  Timer timer;
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_FALSE(timer.has_run_at_least_once());
}

TYPED_TEST(BenchmarkTest, TestTimerStart) {
  Timer timer;
  timer.Start();
  EXPECT_TRUE(timer.initted());
  EXPECT_TRUE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
  timer.Start();
  EXPECT_TRUE(timer.initted());
  EXPECT_TRUE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
  timer.Stop();
  timer.Start();
  EXPECT_TRUE(timer.initted());
  EXPECT_TRUE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
}

TYPED_TEST(BenchmarkTest, TestTimerStop) {
  Timer timer;
  timer.Stop();
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_FALSE(timer.has_run_at_least_once());
  timer.Start();
  timer.Stop();
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
  timer.Stop();
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
}

TYPED_TEST(BenchmarkTest, TestTimerMilliSeconds) {
  Timer timer;
  EXPECT_EQ(timer.MilliSeconds(), 0);
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_FALSE(timer.has_run_at_least_once());
  timer.Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  EXPECT_GE(timer.MilliSeconds(), 300 - kMillisecondsThreshold);
  EXPECT_LE(timer.MilliSeconds(), 300 + kMillisecondsThreshold);
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
}

TYPED_TEST(BenchmarkTest, TestTimerSeconds) {
  Timer timer;
  EXPECT_EQ(timer.Seconds(), 0);
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_FALSE(timer.has_run_at_least_once());
  timer.Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  EXPECT_GE(timer.Seconds(), 0.3 - kMillisecondsThreshold / 1000.f);
  EXPECT_LE(timer.Seconds(), 0.3 + kMillisecondsThreshold / 1000.f);
  EXPECT_TRUE(timer.initted());
  EXPECT_FALSE(timer.running());
  EXPECT_TRUE(timer.has_run_at_least_once());
}

}  // namespace latte