#ifndef LATTE_SYNCEDMEM_H_
#define LATTE_SYNCEDMEM_H_

#include "latte/common.h"

namespace latte {

class SyncedMemory : public Noncopyable {
 public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);
  ~SyncedMemory();

  const void *cpu_data();
  void set_cpu_data(void *data);
  const void *gpu_data();
  void set_gpu_data(void *data);
  void *mutable_cpu_data();
  void *mutable_gpu_data();
  enum class SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() const { return head_; }
  size_t size() const { return size_; }

#ifdef WITH_CUDA
  void async_gpu_push(const cudaStream_t &stream);
#endif

 protected:
  void LatteMallocHost(void **ptr, size_t size, bool *use_cuda);
  void LatteFreeHost(void *ptr, bool use_cuda);

 private:
  void check_device();

  void to_cpu();
  void to_gpu();
  void *cpu_ptr_;
  void *gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int device_;
};

}  // namespace latte

#endif  // LATTE_SYNCEDMEM_H_
