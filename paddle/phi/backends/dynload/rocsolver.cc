/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved. */

#include "paddle/phi/backends/dynload/rocsolver.h"

namespace phi::dynload {

std::once_flag rocsolver_dso_flag;
void *rocsolver_dso_handle;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

ROCSOLVER_ROUTINE_EACH(DEFINE_WRAP);

// AMD ROCm overlay: guard EACH1 -- rocsolver.h only defines
// ROCSOLVER_ROUTINE_EACH1 for HIP_VERSION >= 50300000, so the unguarded
// upstream form fails to compile on older HIP. The guard is compatible with
// upstream's intent on ROCm 6+ and safe on older toolchains.
#ifdef ROCSOLVER_ROUTINE_EACH1
ROCSOLVER_ROUTINE_EACH1(DEFINE_WRAP);
#endif

}  // namespace phi::dynload
