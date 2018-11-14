#include <vector>

#include <gtest/gtest.h>

#include "latte/common.h"
#include "latte/syncedmem.h"
#include "latte/util/math_functions.h"

#include "latte/test/test_latte_main.h"

namespace latte {

class SyncedMemoryTest : public ::testing::Test {};

TEST_F(SyncedMemoryTest, TestInitialization) {
  SyncedMemory mem(10);
  EXPECT_EQ(mem.head(), SyncedMemory::SyncedHead::UNINITIALIZED);
  EXPECT_EQ(mem.size(), 10);
  SyncedMemory *p_mem = new SyncedMemory(10 * sizeof(float));
  EXPECT_EQ(p_mem->size(), 10 * sizeof(float));
  delete p_mem;
}

#ifndef CPU_ONLY

TEST_F(SyncedMemoryTest, TestAllocationCPUGPU) {
  SyncedMemory mem(10);
  EXPECT_TRUE(mem.cpu_data());
  EXPECT_TRUE(mem.gpu_data());
  EXPECT_TRUE(mem.mutable_cpu_data());
  EXPECT_TRUE(mem.mutable_gpu_data());
}

#endif

TEST_F(SyncedMemoryTest, TestAllocationCPU) {
  SyncedMemory mem(10);
  EXPECT_TRUE(mem.cpu_data());
  EXPECT_TRUE(mem.mutable_cpu_data());
}

#ifndef CPU_ONLY

TEST_F(SyncedMemoryTest, TestAllocationGPU) {
  SyncedMemory mem(10);
  EXPECT_TRUE(mem.gpu_data());
  EXPECT_TRUE(mem.mutable_gpu_data());
}

#endif

TEST_F(SyncedMemoryTest, TestCPUWrite) {
  SyncedMemory mem(10);
  void *cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SyncedHead::HEAD_AT_CPU);
  latte_memset(mem.size(), 1, cpu_data);
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(reinterpret_cast<char*>(cpu_data)[i], 1);
  }

  cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SyncedHead::HEAD_AT_CPU);
  latte_memset(mem.size(), 2, cpu_data);
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(reinterpret_cast<char*>(cpu_data)[i], 2);
  }
}

#ifndef CPU_ONLY

TEST_F(SyncedMemoryTest, TestGPURead) {
  SyncedMemory mem(10);
  void *cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SyncedHead::HEAD_AT_CPU);
  latte_memset(mem.size(), 1, cpu_data);
  const void *gpu_data = mem.gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SyncedHead::SYNCED);

  char *recovered_value = new char[10];
  latte_gpu_memcpy(10, gpu_data, recovered_value);
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(recovered_value[i], 1);
  }

  cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SyncedHead::HEAD_AT_CPU);
  latte_memset(mem.size(), 2, cpu_data);
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(reinterpret_cast<char*>(cpu_data)[i], 2);
  }
  gpu_data = mem.gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SyncedHead::SYNCED);
  latte_gpu_memcpy(10, gpu_data, recovered_value);
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(recovered_value[i], 2);
  }
  delete[] recovered_value;
}

TEST_F(SyncedMemoryTest, TestGPUWrite) {
  SyncedMemory mem(10);
  void *gpu_data = mem.mutable_gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SyncedHead::HEAD_AT_GPU);
  latte_gpu_memset(mem.size(), 1, gpu_data);
  const void *cpu_data = mem.cpu_data();
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(reinterpret_cast<const char*>(cpu_data)[i], 1);
  }

  gpu_data = mem.mutable_gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SyncedHead::HEAD_AT_GPU);
  latte_gpu_memset(mem.size(), 2, gpu_data);
  cpu_data = mem.cpu_data();
  for (int i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(reinterpret_cast<const char*>(cpu_data)[i], 2);
  }
  EXPECT_EQ(mem.head(), SyncedMemory::SyncedHead::SYNCED);
}

#endif

}