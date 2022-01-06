# Latte

[![Build Latte](https://github.com/xcnick/latte/actions/workflows/build.yml/badge.svg)](https://github.com/xcnick/latte/actions/workflows/build.yml)

A mini deeplearning framework inspired by [Caffe](https://github.com/BVLC/caffe)

## Dependencies

* GCC
* CUDA
* CUDNN
* CMake >= 3.18

## Build from source

1. Build

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

2. Test(Optional)

```bash
make runtest
```