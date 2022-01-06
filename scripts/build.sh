set -xe

SRC_ROOT=$1
cd ${SRC_ROOT}
cmake -S . -B build
cmake --build build -j