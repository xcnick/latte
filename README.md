# Latte

[![Build Status](https://travis-ci.org/xcnick/latte.svg?branch=master)](https://travis-ci.org/xcnick/latte)

A mini deeplearning framework inspired by [Caffe](https://github.com/BVLC/caffe)

## Dependencies
* GCC
* CUDA
*	CMake

## Build from source
1. Install dependencies.
2. Build
```
mkdir build
cd build && cmake .. && make -j
```
3. Test(Optional)
```
make runtest
```