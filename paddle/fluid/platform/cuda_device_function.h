/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define PADDLE_CUDA_FP16
#include "paddle/fluid/platform/float16.h"
#else
#include <cuda.h>
// NOTE(): support float16 to half in header file.
#define PADDLE_CUDA_FP16
#include <cuda_fp16.h>
#include "paddle/fluid/platform/float16.h"
#endif

#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace platform {

#if CUDA_VERSION < 9000 || defined(PADDLE_WITH_HIP)
#define CREATE_SHFL_MASK(mask, predicate) mask = 0u;
#else
#define FULL_WARP_MASK 0xFFFFFFFF
#define CREATE_SHFL_MASK(mask, predicate) \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#endif

template <typename T>
static __forceinline__ __device__ T CudaShuffleDownSync(unsigned mask, T val,
                                                 int delta, int width = 32) {
#if CUDA_VERSION < 9000 || defined(PADDLE_WITH_HIP)
  return __shfl_down(val, delta, width);
#else
  return __shfl_down_sync(mask, val, static_cast<unsigned>(delta), width);
#endif
}

static __forceinline__ __device__ float CudaShuffleSync(unsigned mask, float val, int src_line,
                                             int width = 32) {
  return __shfl(val, src_line, width);
}

static __forceinline__ __device__ int CudaShuffleDownSync(unsigned mask, int val,
                                                 int delta, int width) {
  return __shfl_down(val, delta, width);
}

static __forceinline__ __device__ int CudaShuffleSync(unsigned mask, int val, int src_line,
                                             int width) {
  return __shfl(val, src_line, width);
}

static __forceinline__ __device__ paddle::platform::float16 CudaShuffleDownSync(unsigned mask, paddle::platform::float16 val,
                                                 int delta, int width) {
  return (float)__shfl_down((float)val, delta, width);
}

static __forceinline__ __device__ paddle::platform::float16 CudaShuffleSync(unsigned mask, paddle::platform::float16 val, int src_line,
                                             int width) {
  return (float)__shfl((float)val, src_line, width);
}

static __forceinline__ __device__ double CudaShuffleDownSync(unsigned mask, double val,
                                                 int delta, int width) {
  return (float)__shfl_down((float)val, delta, width);
}

static __forceinline__ __device__ double CudaShuffleSync(unsigned mask, double val, int src_line,
                                             int width) {
  return (float)__shfl((float)val, src_line, width);
}

template <typename T>
__forceinline__ __device__ T CudaShuffleSync(unsigned mask, T val, int src_line,
                                             int width = 32) {
#if CUDA_VERSION < 9000
  return __shfl(val, src_line, width);
#else
  return __shfl_sync(mask, val, src_line, width);
#endif
}

template <typename T>
HOSTDEVICE T Infinity() {
  return INFINITY;
}

#ifdef PADDLE_WITH_HIP
template <typename T>
__device__ T reduceSum(T val, int tid, int len) {
  const int warpSize = 32;
  __shared__ T shm[warpSize*warpSize];
  shm[tid] = val;

  __syncthreads();

  if (tid == 0 ) {
    for (int i = 1 ; i < len ; i++)
      val += shm[i];
  }

  return val;
}
#else
template <typename T>
__device__ T reduceSum(T val, int tid, int len) {
  // NOTE(zcd): The warp size should be taken from the
  // parameters of the GPU but not specified as 32 simply.
  // To make the reduceSum more efficiently,
  // I use Warp-Level Parallelism and assume the Warp size
  // is 32 which may be different for different GPU,
  // but most card's warp size is 32.
  const int warpSize = 32;
  __shared__ T shm[warpSize];
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < len);

  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += platform::CudaShuffleDownSync(mask, val, offset);

  if (tid < warpSize) shm[tid] = 0;
  __syncthreads();

  if (tid % warpSize == 0) {
    shm[tid / warpSize] = val;
  }
  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);

  if (tid < warpSize) {
    val = shm[tid];
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      val += platform::CudaShuffleDownSync(mask, val, offset);
  }
  return val;
}
#endif

}  // namespace platform
}  // namespace paddle
