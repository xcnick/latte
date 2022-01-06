#ifndef LATTE_UTIL_BENCHMARK_H_
#define LATTE_UTIL_BENCHMARK_H_

#include <chrono>

#include "latte/util/device_alternate.h"

namespace latte {

class Timer {
 public:
  Timer();
  virtual ~Timer();
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();
  virtual float Seconds();

  inline bool initted() { return initted_; }
  inline bool running() { return running_; }
  inline bool has_run_at_least_once() { return has_run_at_least_once_; }

 protected:
  void Init();

  bool initted_;
  bool running_;
  bool has_run_at_least_once_;
#ifdef WITH_CUDA
  cudaEvent_t start_gpu_;
  cudaEvent_t stop_gpu_;
#endif
  using clock = std::chrono::high_resolution_clock;
  using microseconds = std::chrono::microseconds;
  using milliseconds = std::chrono::milliseconds;

  clock::time_point start_cpu_;
  clock::time_point stop_cpu_;
  float elapsed_milliseconds_;
  float elapsed_microseconds_;
};

class CPUTimer : public Timer {
public:
  explicit CPUTimer();
  virtual ~CPUTimer() {}
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();
};

}  // namespace latte

#endif  // LATTE_UTIL_BENCHMARK_H_