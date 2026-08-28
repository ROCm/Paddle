# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

if(NOT WITH_ROCM)
  return()
endif()

if(NOT DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH
      "/opt/rocm"
      CACHE PATH "Path to which ROCm has been installed")
  set(HIP_PATH
      ${ROCM_PATH}/hip
      CACHE PATH "Path to which HIP has been installed")
  set(HIP_CLANG_PATH
      ${ROCM_PATH}/llvm/bin
      CACHE PATH "Path to which clang has been installed")
else()
  set(ROCM_PATH
      $ENV{ROCM_PATH}
      CACHE PATH "Path to which ROCm has been installed")
  set(HIP_PATH
      ${ROCM_PATH}/hip
      CACHE PATH "Path to which HIP has been installed")
  set(HIP_CLANG_PATH
      ${ROCM_PATH}/llvm/bin
      CACHE PATH "Path to which clang has been installed")
endif()
if(EXISTS "${HIP_PATH}/cmake")
  set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
else()
  set(CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake/hip" ${CMAKE_MODULE_PATH})
endif()
set(CMAKE_PREFIX_PATH "${ROCM_PATH}" ${CMAKE_PREFIX_PATH})

find_package(HIP REQUIRED)
include_directories(${ROCM_PATH}/include)
message(STATUS "HIP version: ${HIP_VERSION}")
message(STATUS "HIP_CLANG_PATH: ${HIP_CLANG_PATH}")

macro(find_hip_version hip_header_file)
  file(READ ${hip_header_file} HIP_VERSION_FILE_CONTENTS)

  string(REGEX MATCH "define HIP_VERSION_MAJOR +([0-9]+)" HIP_MAJOR_VERSION
               "${HIP_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define HIP_VERSION_MAJOR +([0-9]+)" "\\1"
                       HIP_MAJOR_VERSION "${HIP_MAJOR_VERSION}")
  string(REGEX MATCH "define HIP_VERSION_MINOR +([0-9]+)" HIP_MINOR_VERSION
               "${HIP_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define HIP_VERSION_MINOR +([0-9]+)" "\\1"
                       HIP_MINOR_VERSION "${HIP_MINOR_VERSION}")
  string(REGEX MATCH "define HIP_VERSION_PATCH +([0-9]+)" HIP_PATCH_VERSION
               "${HIP_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define HIP_VERSION_PATCH +([0-9]+)" "\\1"
                       HIP_PATCH_VERSION "${HIP_PATCH_VERSION}")

  if(NOT HIP_MAJOR_VERSION)
    set(HIP_VERSION "???")
    message(
      WARNING "Cannot find HIP version in ${HIP_PATH}/include/hip/hip_version.h"
    )
  else()
    math(
      EXPR
      HIP_VERSION
      "${HIP_MAJOR_VERSION} * 10000000 + ${HIP_MINOR_VERSION} * 100000   + ${HIP_PATCH_VERSION}"
    )
    # ROCM_GE_6 = "ROCm 6 or newer" (true for every ROCm >= 6, including the
    # ROCm 10 build floor). It is NOT "exactly ROCm 6"; guards below read as
    # "on ROCm 6+" not "on ROCm 6 only".
    if(HIP_VERSION GREATER 60000000)
      set(ROCM_GE_6 ON)
    endif()
    message(
      STATUS
        "Current HIP header is ${HIP_PATH}/include/hip/hip_version.h "
        "Current HIP version is v${HIP_MAJOR_VERSION}.${HIP_MINOR_VERSION}.${HIP_PATCH_VERSION}. "
    )
  endif()
endmacro()
if(EXISTS "${HIP_PATH}/include/hip")
  find_hip_version(${HIP_PATH}/include/hip/hip_version.h)
else()
  find_hip_version(${ROCM_PATH}/include/hip/hip_version.h)
endif()

macro(find_package_and_include PACKAGE_NAME)
  find_package("${PACKAGE_NAME}" REQUIRED)
  if(EXISTS "${ROCM_PATH}/${PACKAGE_NAME}/include")
    include_directories("${ROCM_PATH}/${PACKAGE_NAME}/include")
  else()
    include_directories("${ROCM_PATH}/include/${PACKAGE_NAME}")
  endif()
  message(STATUS "${PACKAGE_NAME} version: ${${PACKAGE_NAME}_VERSION}")
endmacro()

if(ROCM_GE_6)
  message(STATUS "Current HIP is ROCm 6 or newer")
endif()
find_package_and_include(miopen)
find_package_and_include(rocblas)
find_package_and_include(hipblaslt)
find_package_and_include(hiprand)
find_package_and_include(rocrand)
find_package_and_include(rccl)
find_package_and_include(rocthrust)
find_package_and_include(hipcub)
find_package_and_include(rocprim)
find_package_and_include(hipsparse)
find_package_and_include(rocsparse)
find_package_and_include(rocfft)
find_package_and_include(rocsolver)

if(CCACHE_PATH)
  set(HIP_HIPCC_EXECUTABLE ${CCACHE_PATH} ${HIP_HIPCC_EXECUTABLE})
endif()

# set CXX flags for HIP
set(CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} -D__HIP_PLATFORM_HCC__ -D__HIP_PLATFORM_AMD__ -DROCM_NO_WRAPPER_HEADER_WARNING"
)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_HCC__ -D__HIP_PLATFORM_AMD__ -DROCM_NO_WRAPPER_HEADER_WARNING"
)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP")
set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_HIP)

# On ROCm 6+, rocThrust bundles CCCL/libcudacxx. thrust/complex.h (included by
# paddle/phi/common/complex.h) pulls in cuda/std/cmath half/bf16 math overloads
# that reference intrinsics (__hisnan, __hisinf, __habs, __hmax,
# __double2bfloat16) only declared in the HIP fp16/bf16 headers under
# `defined(__clang__) && defined(__HIP__)`. A plain host C++ compile has neither,
# so these overloads are undeclared -- and even a Clang C++ (non-HIP) compile
# fails on host-only __half<->__hip_bfloat16 conversion ambiguities in the CCCL
# headers. Setting __HIP_NO_HALF_CONVERSIONS__/__HIP_NO_HALF_OPERATORS__ does not
# help: the CCCL fp16/bf16 path is force-enabled by __HIP_PLATFORM_AMD__ and its
# intrinsics simply are not visible to a host compile. The one compilation mode
# that accepts thrust/complex.h on this toolchain is Clang's HIP language mode
# (-x hip), which makes the fp16/bf16 intrinsics available and resolves the
# overloads while leaving ordinary host C++ unaffected. Compile the host C++ side
# of the ROCm build in HIP language mode so host translation units that
# transitively include thrust build. This requires a Clang toolchain: configure
# the ROCm build with the ROCm clang++ as the C/C++ compiler
# (-DCMAKE_C_COMPILER=<rocm>/llvm/bin/clang -DCMAKE_CXX_COMPILER=.../clang++).
#
# Do NOT put `-x hip` in the global CMAKE_CXX_FLAGS. CMake's configure-time
# check_cxx_symbol_exists / check_type_size / compiler-ABI try-compiles reuse
# CMAKE_CXX_FLAGS; under `-x hip` they treat the intermediate as HIP source and
# default to gfx906, so e.g. cmake/flags.cmake's UINT64_MAX probe fails and
# aborts configure. It would also leak into ExternalProjects that capture
# ${CMAKE_CXX_FLAGS} (protobuf, onednn), which are ordinary host C++ and should
# not be compiled in HIP language mode. Instead, record the intent here and let
# the top-level CMakeLists apply it via add_compile_options AFTER all
# configure-time check_*() calls (i.e. after include(flags)) and after the
# third_party externals are declared, so only Paddle's own targets (added via
# add_subdirectory(paddle)) receive the flag. The guard no-ops on a non-Clang
# compiler so a CPU/GCC configure is safe.
# Host C++ no longer needs HIP language mode: phi/common/complex.h now guards
# thrust/complex.h behind device compilation, so host .cc TUs do not pull
# rocThrust/CCCL and compile as plain C++. Keeping host OUT of `-x hip` is what
# stops __HIPCC__ leaking onto host code (the enforce.h/PADDLE_ENFORCE break in
# cf_op.cc and ~100 siblings). Default OFF; opt back in with
# -DPADDLE_HOST_HIP_LANGUAGE=ON if a host TU is found to still need `-x hip`.
option(PADDLE_HOST_HIP_LANGUAGE
       "Compile Paddle host C++ in Clang HIP language mode (-x hip)" OFF)

# define HIP_CXX_FLAGS
list(APPEND HIP_CXX_FLAGS -fPIC)
# ROCm thrust-compat: force-include the compat shim FIRST into every HIP TU so
# rocThrust's internal algo cross-references (e.g. copy_if.h -> thrust::inclusive_scan)
# resolve, pre-empting the "no member named '<algo>' in namespace 'thrust'" class.
list(APPEND HIP_CXX_FLAGS -include ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/thrust_compat.h)
list(APPEND HIP_CXX_FLAGS -D__HIP_PLATFORM_HCC__=1)
list(APPEND HIP_CXX_FLAGS -D__HIP_PLATFORM_AMD__=1)
# Note(qili93): HIP has compile conflicts of float16.h as platform::float16 overload std::is_floating_point and std::is_integer
list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_CONVERSIONS__=1)
list(APPEND HIP_CXX_FLAGS -DROCM_NO_WRAPPER_HEADER_WARNING)
list(APPEND HIP_CXX_FLAGS -Wno-macro-redefined)
list(APPEND HIP_CXX_FLAGS -Wno-inconsistent-missing-override)
list(APPEND HIP_CXX_FLAGS -Wno-exceptions)
list(APPEND HIP_CXX_FLAGS -Wno-shift-count-negative)
list(APPEND HIP_CXX_FLAGS -Wno-shift-count-overflow)
list(APPEND HIP_CXX_FLAGS -Wno-unused-command-line-argument)
list(APPEND HIP_CXX_FLAGS -Wno-duplicate-decl-specifier)
list(APPEND HIP_CXX_FLAGS -Wno-implicit-int-float-conversion)
list(APPEND HIP_CXX_FLAGS -Wno-pass-failed)
list(APPEND HIP_CXX_FLAGS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
list(APPEND HIP_CXX_FLAGS -Wno-unused-result)
list(APPEND HIP_CXX_FLAGS -Wno-deprecated-declarations)
list(APPEND HIP_CXX_FLAGS -Wno-format)
list(APPEND HIP_CXX_FLAGS -Wno-dangling-gsl)
list(APPEND HIP_CXX_FLAGS -Wno-unused-value)
list(APPEND HIP_CXX_FLAGS -Wno-braced-scalar-init)
list(APPEND HIP_CXX_FLAGS -Wno-return-type)
list(APPEND HIP_CXX_FLAGS -Wno-pragma-once-outside-header)
list(APPEND HIP_CXX_FLAGS -Wno-deprecated-builtins)
list(APPEND HIP_CXX_FLAGS -Wno-switch)
list(APPEND HIP_CXX_FLAGS -Wno-literal-conversion)
list(APPEND HIP_CXX_FLAGS -Wno-constant-conversion)
list(APPEND HIP_CXX_FLAGS -Wno-defaulted-function-deleted)
list(APPEND HIP_CXX_FLAGS -Wno-sign-compare)
list(APPEND HIP_CXX_FLAGS -Wno-bitwise-instead-of-logical)
list(APPEND HIP_CXX_FLAGS -Wno-unknown-warning-option)
list(APPEND HIP_CXX_FLAGS -Wno-unused-lambda-capture)
list(APPEND HIP_CXX_FLAGS -Wno-unused-variable)
list(APPEND HIP_CXX_FLAGS -Wno-unused-but-set-variable)
list(APPEND HIP_CXX_FLAGS -Wno-reorder-ctor)
list(APPEND HIP_CXX_FLAGS -Wno-deprecated-copy-with-user-provided-copy)
list(APPEND HIP_CXX_FLAGS -Wno-unused-local-typedef)
list(APPEND HIP_CXX_FLAGS -Wno-missing-braces)
list(APPEND HIP_CXX_FLAGS -Wno-sometimes-uninitialized)
list(APPEND HIP_CXX_FLAGS -Wno-deprecated-copy)
list(APPEND HIP_CXX_FLAGS -Wno-pessimizing-move)
list(APPEND HIP_CXX_FLAGS -std=c++17)
list(APPEND HIP_CXX_FLAGS --gpu-max-threads-per-block=1024)

if(CMAKE_BUILD_TYPE MATCHES Debug)
  list(APPEND HIP_CXX_FLAGS -g2)
  list(APPEND HIP_CXX_FLAGS -O0)
  list(APPEND HIP_HIPCC_FLAGS -fdebug-info-for-profiling)
endif()

set(HIP_HCC_FLAGS ${HIP_CXX_FLAGS})
set(HIP_CLANG_FLAGS ${HIP_CXX_FLAGS})
# Ask hcc to generate device code during compilation so we can use
# host linker to link.
list(APPEND HIP_HCC_FLAGS -fno-gpu-rdc)
if(ROCM_GE_6)
  # AMD ROCm overlay: target AMD Instinct gfx942 on ROCm 6+.
  list(APPEND HIP_HCC_FLAGS --offload-arch=gfx942)
else()
  list(APPEND HIP_HCC_FLAGS --offload-arch=gfx906) # Z100 (ZIFANG)
  list(APPEND HIP_HCC_FLAGS --offload-arch=gfx926) # K100 (KONGING)
  list(APPEND HIP_HCC_FLAGS --offload-arch=gfx928) # K100_AI (KONGING_AI)
  list(APPEND HIP_HCC_FLAGS --offload-arch=gfx936) # BW1000 (BOWEN)
endif()
list(APPEND HIP_CLANG_FLAGS -fno-gpu-rdc)
if(ROCM_GE_6)
  # AMD ROCm overlay: target AMD Instinct gfx942 on ROCm 6+.
  list(APPEND HIP_CLANG_FLAGS --offload-arch=gfx942)
else()
  list(APPEND HIP_CLANG_FLAGS --offload-arch=gfx906) # Z100 (ZIFANG)
  list(APPEND HIP_CLANG_FLAGS --offload-arch=gfx926) # K100 (KONGING)
  list(APPEND HIP_CLANG_FLAGS --offload-arch=gfx928) # K100_AI (KONGING_AI)
  list(APPEND HIP_CLANG_FLAGS --offload-arch=gfx936) # BW1000 (BOWEN)
endif()

# CMake >= 4.x's HIP language support treats an empty CMAKE_HIP_ARCHITECTURES as a
# hard generate-time error for any target with HIP sources. The offload-arch
# compiler flags above are not sufficient for that check, so set the property from
# the same architecture list so a bare "cmake .." configures on modern CMake
# without the caller passing -DCMAKE_HIP_ARCHITECTURES.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
  if(ROCM_GE_6)
    set(CMAKE_HIP_ARCHITECTURES "gfx942")
  else()
    set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx926;gfx928;gfx936")
  endif()
endif()
message(STATUS "CMAKE_HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")


if(HIP_COMPILER STREQUAL clang)
  set(hip_library_name amdhip64)
else()
  set(hip_library_name hip_hcc)
endif()
message(STATUS "HIP library name: ${hip_library_name}")

# set HIP link libs
if(ROCM_GE_6)
  # AMD ROCm overlay: on ROCm 6+ the HIP runtime lib and the clang toolchain
  # live under ${ROCM_PATH} (do NOT hardcode /opt/rocm -- the manylinux ROCm
  # base installs the toolkit elsewhere). Only set the compiler if it was not
  # already selected by the toolchain / cache.
  find_library(ROCM_HIPRTC_LIB ${hip_library_name} HINTS ${ROCM_PATH}/lib
                                                         ${HIP_PATH}/lib)

  if(NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "${HIP_CLANG_PATH}/clang")
  endif()
  if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${HIP_CLANG_PATH}/clang++")
  endif()
else()
  find_library(ROCM_HIPRTC_LIB ${hip_library_name} HINTS ${HIP_PATH}/lib)
endif()
message(STATUS "ROCM_HIPRTC_LIB: ${ROCM_HIPRTC_LIB}")

include(thrust)

if(WITH_Z100)
  add_definitions(-DCINN_WITH_Z100)
endif()
