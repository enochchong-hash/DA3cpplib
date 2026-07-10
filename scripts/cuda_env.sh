#!/bin/bash
# cuda_env.sh - Locate a usable CUDA toolkit and put it first on PATH.
# Sourced by setup.sh and build.sh; not meant to be executed directly.
#
# Why this exists: Ubuntu's 'nvidia-cuda-toolkit' apt package installs an
# ancient nvcc (11.5) at /usr/bin/nvcc, which shadows a proper NVIDIA toolkit
# installed at /usr/local/cuda. Prefer /usr/local/cuda (then any versioned
# /usr/local/cuda-12*/13* directory) so builds always see the real toolkit
# regardless of PATH pollution. Sets DA3_CUDA_HOME when a toolkit is found.

da3_cuda_home=""
for dir in /usr/local/cuda /usr/local/cuda-13* /usr/local/cuda-12*; do
  [ -x "$dir/bin/nvcc" ] || continue
  da3_cuda_home="$dir"
  break
done
if [ -n "$da3_cuda_home" ]; then
  export PATH="$da3_cuda_home/bin:$PATH"
  export DA3_CUDA_HOME="$da3_cuda_home"
fi
unset da3_cuda_home
