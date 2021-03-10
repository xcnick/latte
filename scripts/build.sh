set -xe

SRC_ROOT=$1
mkdir ${SRC_ROOT}/build
cd ${SRC_ROOT}/build
cmake ..
cmake --build . -- -j