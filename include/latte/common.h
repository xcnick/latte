#ifndef LATTE_COMMON_H_
#define LATTE_COMMON_H_

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <climits>
#include <cmath>
#include <fstream>   // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>

#include "latte/util/device_alternate.h"

// Convert macro to string
#define STRINGIFY(m) #m
#define AS_STRING(m) #m

namespace latte {

#define INSTANTIATE_CLASS(classname) \
  template class classname<float>;   \
  template class classname<double>

#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
  template void classname<float>::Forward_gpu(   \
      const std::vector<Blob<float> *> &bottom,  \
      const std::vector<Blob<float> *> &top);    \
  template void classname<double>::Forward_gpu(  \
      const std::vector<Blob<double> *> &bottom, \
      const std::vector<Blob<double> *> &top)

#define INSTANTIATE_LAYER_GPU_FUNCS(classname) \
  INSTANTIATE_LAYER_GPU_FORWARD(classname)

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

class Noncopyable {
 protected:
  constexpr Noncopyable() = default;
  ~Noncopyable() = default;
  Noncopyable(const Noncopyable &) = delete;
  Noncopyable(Noncopyable &&) = delete;
  Noncopyable &operator=(const Noncopyable &) = delete;
  Noncopyable &operator=(Noncopyable &) = delete;
};

// Common functions and classes from std that latte often uses.
using std::fstream;
using std::ios;
using std::isinf;
using std::isnan;
using std::iterator;
using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::shared_ptr;
using std::string;
using std::stringstream;
using std::vector;

// A global initialization function that you should call in your main function.
// Currently it initializes google flags and google logging.
void GlobalInit(int *pargc, char ***pargv);

inline void string_split(vector<string> *tokens, const string &str,
                         const string &delimiters) {
  // Skip delimiters at beginning.
  string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  string::size_type pos = str.find_first_of(delimiters, lastPos);
  while (string::npos != pos || string::npos != lastPos) {
    // Found a token, add it to the vector.
    tokens->push_back(str.substr(lastPos, pos - lastPos));
    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);
    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }
}

class Latte : public Noncopyable {
 public:
  ~Latte();

  static Latte &Get();

  enum Brew { CPU, GPU };

  // This random number generator facade hides boost and CUDA rng
  // implementation from one another (for cross-platform compatibility).
  class RNG {
   public:
    RNG();
    explicit RNG(unsigned int seed);
    explicit RNG(const RNG &);
    RNG &operator=(const RNG &);
    void *generator();

   private:
    class Generator;
    shared_ptr<Generator> generator_;
  };

  // Getters for boost rng, curand, and cublas handles
  inline static RNG &rng_stream() {
    if (!Get().random_generator_) {
      Get().random_generator_.reset(new RNG());
    }
    return *(Get().random_generator_);
  }
#ifdef WITH_CUDA
  inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
  inline static curandGenerator_t curand_generator() {
    return Get().curand_generator_;
  }
#endif

  // Returns the mode: running on CPU or GPU.
  inline static Brew mode() { return Get().mode_; }
  // The setters for the variables
  // Sets the mode. It is recommended that you don't change the mode halfway
  // into the program since that may cause allocation of pinned memory being
  // freed in a non-pinned way, which may cause problems - I haven't verified
  // it personally but better to note it here in the header file.
  inline static void set_mode(Brew mode) { Get().mode_ = mode; }
  // Sets the random seed of both boost and curand
  static void set_random_seed(const unsigned int seed);
  // Sets the device. Since we have cublas and curand stuff, set device also
  // requires us to reset those values.
  static void SetDevice(const int device_id);
  // Prints the current GPU status.
  static void DeviceQuery();
  // Check if specified device is available
  static bool CheckDevice(const int device_id);
  // Search from start_id to the highest possible device ordinal,
  // return the ordinal of the first available device.
  static int FindDevice(const int start_id = 0);

 protected:
#ifdef WITH_CUDA
  cublasHandle_t cublas_handle_;
  curandGenerator_t curand_generator_;
#endif
  shared_ptr<RNG> random_generator_;

  Brew mode_;

 private:
  Latte();
};

}  // namespace latte

#endif  // LATTE_COMMON_H_