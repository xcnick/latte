#include "latte/syncedmem.h"
#include <cstdlib>
#include "latte/util/math_functions.h"

namespace latte {

SyncedMemory::SyncedMemory()
    : cpu_ptr_(nullptr),
      gpu_ptr_(nullptr),
      size_(0),
      head_(SyncedHead::UNINITIALIZED),
      own_cpu_data_(false),
      cpu_malloc_use_cuda_(false),
      own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device));
#endif
#endif
}

SyncedMemory::SyncedMemory(size_t size)
    : cpu_ptr_(nullptr),
      gpu_ptr_(nullptr),
      size_(size),
      head_(SyncedHead::UNINITIALIZED),
      own_cpu_data_(false),
      cpu_malloc_use_cuda_(false),
      own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device));
#endif
#endif
}

SyncedMemory::~SyncedMemory() {
  check_device();
  if (cpu_ptr_ && own_cpu_data_) {
    LatteFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
#endif
}

void SyncedMemory::LatteMallocHost(void **ptr, size_t size, bool *use_cuda) {
#ifndef CPU_ONLY
  if (Latte::mode() == Latte::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif

  *ptr = std::malloc(size);
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

void SyncedMemory::LatteFreeHost(void *ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
  free(ptr);
}

void SyncedMemory::check_device() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  cudaGetDevice(&device);
  CHECK(device == device_);
  if (gpu_ptr_ && own_gpu_data_) {
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerAttributes(&attributes, gpu_ptr_));
    CHECK(attributes.device == device_);
  }
#endif
#endif
}

void SyncedMemory::to_cpu() {
  check_device();
  switch (head_) {
    case SyncedHead::UNINITIALIZED:
      LatteMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      latte_memset(size_, 0, cpu_ptr_);
      head_ = SyncedHead::HEAD_AT_CPU;
      own_cpu_data_ = true;
      break;
    case SyncedHead::HEAD_AT_GPU:
#ifndef CPU_ONLY
      if (cpu_ptr_ == nullptr) {
        LatteMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
        own_cpu_data_ = true;
      }
      latte_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
      head_ = SyncedHead::SYNCED;
      break;
#else
      NO_GPU;
#endif
      break;
    case SyncedHead::HEAD_AT_CPU:
    case SyncedHead::SYNCED:
      break;
  }
}

void SyncedMemory::to_gpu() {
  check_device();
#ifndef CPU_ONLY
  switch (head_) {
    case SyncedHead::UNINITIALIZED:
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      latte_gpu_memset(size_, 0, gpu_ptr_);
      head_ = SyncedHead::HEAD_AT_GPU;
      own_gpu_data_ = true;
      break;
    case SyncedHead::HEAD_AT_CPU:
      if (gpu_ptr_ == nullptr) {
        CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
        own_gpu_data_ = true;
      }
      latte_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
      head_ = SyncedHead::SYNCED;
      break;
    case SyncedHead::HEAD_AT_GPU:
    case SyncedHead::SYNCED:
      break;
  }
#else
  NO_GPU;
#endif
}

const void *SyncedMemory::cpu_data() {
  check_device();
  to_cpu();
  return reinterpret_cast<const void *>(cpu_ptr_);
}

void SyncedMemory::set_cpu_data(void *data) {
  check_device();
  CHECK(data);
  if (own_cpu_data_) {
    LatteFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = SyncedHead::HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void *SyncedMemory::gpu_data() {
  check_device();
#ifndef CPU_ONLY
  to_gpu();
  return reinterpret_cast<const void *>(gpu_ptr_);
#else
  NO_GPU;
  return nullptr;
#endif
}

void SyncedMemory::set_gpu_data(void *data) {
  check_device();
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = SyncedHead::HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

void *SyncedMemory::mutable_cpu_data() {
  check_device();
  to_cpu();
  head_ = SyncedHead::HEAD_AT_CPU;
  return cpu_ptr_;
}

void *SyncedMemory::mutable_gpu_data() {
  check_device();
#ifndef CPU_ONLY
  to_gpu();
  head_ = SyncedHead::HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return nullptr;
#endif
}

#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t &stream) {
  check_device();
  CHECK(head_ == SyncedHead::HEAD_AT_CPU);
  if (gpu_ptr_ == nullptr) {
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  head_ = SyncedHead::SYNCED;
}
#endif

}  // namespace latte