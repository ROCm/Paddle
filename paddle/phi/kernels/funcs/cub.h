// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
// ROCm 10 hipCUB ships arg_index/cache_modified/tex_obj iterators but NOT
// Counting/TransformInputIterator, which Paddle uses via the `cub` alias
// (reduce_function.h:975, squared_l2_norm.h, top_k_*.h). thrust DOES provide
// equivalents (present at /opt/rocm/include/thrust/iterator/), so shim them into the
// hipcub namespace as thrust aliases -- one place, every cub-iterator site resolves.
// (precedent: shuffle_batch.cu.h already uses thrust::transform_iterator on ROCm.)
#include <hipcub/hipcub_version.hpp>
// hipCUB >= 4.6.0 (ROCm 10.1, HIPCUB_VERSION 400600) ships its own
// Counting/TransformInputIterator in the hipcub namespace, so defining the thrust
// shims below would be an ambiguous redefinition. Only shim on older hipCUB that
// lacks them.
#if !defined(HIPCUB_VERSION) || (HIPCUB_VERSION < 400600)
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
namespace hipcub {
template <typename T>
using CountingInputIterator = thrust::counting_iterator<T>;
template <typename ValueType, typename ConversionOp, typename InputIterator>
using TransformInputIterator = thrust::transform_iterator<ConversionOp, InputIterator>;
}  // namespace hipcub
#endif  // HIPCUB_VERSION < 400600
namespace cub = hipcub;
#endif
