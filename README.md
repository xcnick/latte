# Latte

[![Build Latte](https://github.com/xcnick/latte/actions/workflows/build.yml/badge.svg)](https://github.com/xcnick/latte/actions/workflows/build.yml)

A mini deeplearning framework inspired by [Caffe](https://github.com/BVLC/caffe)

## Highlight

Compared with [Caffe](https://github.com/BVLC/caffe), Latte has the following features:

- Only support forward computation for inference
- Using Modern CMake based build system
- Managing third-party libraries using CMake
- Simplify dependencies, remove boost, etc.

## Dependencies

* GCC
* CUDA
* cuDNN
* CMake >= 3.18

## Build from source

1. Build docker development image

```bash
docker build -t latte:dev -f docker/Dockerfile .
```

2. Build

```bash
cmake -S . -B build -G Ninja
cmake --build build
```
