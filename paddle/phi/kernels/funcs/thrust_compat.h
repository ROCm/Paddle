// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// ROCm thrust-compat shim -- FORCE-INCLUDED into every HIP translation unit via the
// ROCm build's HIP_CXX_FLAGS (cmake/hip.cmake), so it lands before any Paddle/thrust
// code. Rationale: rocThrust/CCCL-on-HIP headers carry INTERNAL cross-references that
// assume the referenced algorithm header is already included in the current TU -- e.g.
// thrust/system/hip/detail/copy_if.h calls thrust::inclusive_scan without itself
// including thrust/scan.h, so a TU that pulls thrust::copy_if fails with
// "no member named 'inclusive_scan' in namespace 'thrust'". Pre-including the common
// thrust algorithm + iterator headers here resolves those cross-references and pre-empts
// the whole "no member named '<algo>' in namespace 'thrust'" class in ONE place, instead
// of chasing them per-TU. No-op on CUDA (NVCC thrust does not have this issue) and on a
// plain host C++ compile.
#if defined(__HIPCC__)
#include <thrust/adjacent_difference.h>
#include <thrust/complex.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

// ROCm 10.1: rocPRIM radix sort requires a `float_bit_mask` trait for any
// floating-point key type (phi::float16 / phi::bfloat16 specialize
// std::is_floating_point, so rocPRIM treats them as floating-point and demands the
// bit layout; it FORBIDS re-declaring is_arithmetic/number_format for them). This
// registration must be visible in EVERY TU that radix-sorts these types (argsort,
// top_k, mode, ...), so it lives here in the force-included compat header -- one
// global definition instead of per-kernel duplicates that miss TUs like mode_kernel.
#include <rocprim/type_traits.hpp>
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
template <>
struct rocprim::traits::define<phi::dtype::float16> {
  using float_bit_mask =
      rocprim::traits::float_bit_mask::values<uint16_t, 0x8000, 0x7C00, 0x03FF>;
};
template <>
struct rocprim::traits::define<phi::dtype::bfloat16> {
  using float_bit_mask =
      rocprim::traits::float_bit_mask::values<uint16_t, 0x8000, 0x7F80, 0x007F>;
};
#endif  // __HIPCC__
