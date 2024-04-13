#!/bin/bash
set -e
mkdir -p /home/PaddleCustomDevice/backends/musa/build
cd /home/PaddleCustomDevice/backends/musa/build
cmake .. -DWITH_MKLDNN=ON && make -j128
pip install /home/PaddleCustomDevice/backends/musa/build/dist/paddle_custom_musa-0.0.0-cp39-cp39-linux_x86_64.whl --force-reinstall
