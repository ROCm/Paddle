/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
B
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>
#include "paddle/platform/dynload/cudnn.h"
#include "paddle/platform/enforce.h"
#include "paddle/platform/macros.h"
#include "miopen/miopen.h"

namespace paddle {
namespace platform {

#define CUDNN_ENFORCE(condition)                                  \
  do {                                                            \
    miopenStatus_t status = condition;                             \
    if (status != miopenStatusSuccess) {                         \
      PADDLE_THROW("miopen call failed");                          \
    }                                                             \
  } while (false)

enum class DataLayout {  // Not use
  kNHWC,
  kNCHW,
  kNCDHW,
  kNCHW_VECT_C,
};

enum class PoolingMode {

  kMaximum,
  kAverage,
};

template <typename T>
class CudnnDataType;

template <>
class CudnnDataType<float> {
 public:

  static const miopenDataType_t type = miopenFloat;
  typedef const float ScalingParamType;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};


template <>
class CudnnDataType<double> {
 public:
  //static const miopenDataType_t type = CUDNN_DATA_DOUBLE;
  typedef const double ScalingParamType;
  static ScalingParamType* kOne() {

    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

//inline cudnnTensorFormat_t GetCudnnTensorFormat(
//    const DataLayout& order) {  // Not use
//  switch (order) {
//    case DataLayout::kNHWC:
//      return CUDNN_TENSOR_NHWC;
//    case DataLayout::kNCHW:
//      return CUDNN_TENSOR_NCHW;
//    case DataLayout::kNCDHW:
//      return CUDNN_TENSOR_NCHW;  // NOTE: cudnn treat NdTensor as the same
//    default:
//      PADDLE_THROW("Unknown cudnn equivalent for order");
//  }
//  return CUDNN_TENSOR_NCHW;
//}

class ScopedTensorDescriptor {
 public:

  ScopedTensorDescriptor() {
    PADDLE_ENFORCE(dynload::miopenCreateTensorDescriptor(&desc_));
  }
  ~ScopedTensorDescriptor() {
    PADDLE_ENFORCE(dynload::miopenDestroyTensorDescriptor(desc_));
  }

 inline miopenTensorDescriptor_t descriptor(/*const cudnnTensorFormat_t format,*/
                                           const miopenDataType_t type,
                                           const std::vector<int>& dims,
                                           const int groups = 1) {
   // the format is not used now, will add later
   std::vector<int> strides(dims.size());
   strides[dims.size() - 1] = 1;
   for (int i = dims.size() - 2; i >= 0; i--) {
     strides[i] = dims[i + 1] * strides[i + 1];
   }
   // Update tensor descriptor dims setting if groups > 1
   // NOTE: Assume using NCHW or NCDHW order
   std::vector<int> dims_with_group(dims.begin(), dims.end());  // copy
   if (groups > 1) {
     dims_with_group[1] = dims_with_group[1] / groups;
   }
   if (dims_with_group.size()!=4){
   	PADDLE_THROW("miopen only supports 4D tensors, dim=%d not allowed",dims_with_group.size());
   }
   PADDLE_ENFORCE(dynload::miopenSet4dTensorDescriptor(
       desc_, type, dims_with_group[0], dims_with_group[1], dims_with_group[2], dims_with_group[3]));
   return desc_;
 }

  template <typename T>
  inline miopenTensorDescriptor_t descriptor(const DataLayout& order,
                                            const std::vector<int>& dims,
                                            const int groups = 1) {
    return descriptor(/*GetCudnnTensorFormat(order),*/ CudnnDataType<T>::type, dims,
                      groups);
  }

 private:
  miopenTensorDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedTensorDescriptor);
};

class ScopedFilterDescriptor {
 public:
  ScopedFilterDescriptor() {
    PADDLE_ENFORCE(dynload::miopenCreateTensorDescriptor(&desc_));
  }
  ~ScopedFilterDescriptor() {
    PADDLE_ENFORCE(dynload::miopenDestroyTensorDescriptor(desc_));
  }
  inline miopenTensorDescriptor_t descriptor(/*const cudnnTensorFormat_t format,*/
                                            const miopenDataType_t type,
                                            const std::vector<int>& kernel,
                                            const int groups = 1) {
    // filter layout: MCHW(MCDHW), where M is the number of
    // output image channels, C is the number of input image channels,
    // D is the depth of the filter, H is the height of the filter, and W is the
    // width of the filter.
    std::vector<int> kernel_with_group(kernel.begin(), kernel.end());
    if (groups > 1) {
      kernel_with_group[0] /= groups;
      // NOTE: input filter(C) of the filter is already asserted to be C/groups.
    }
    if (kernel_with_group.size()!=4){
        PADDLE_THROW("miopen only supports 4D filters, dim=%d not allowed",kernel_with_group.size());
    }
    PADDLE_ENFORCE(dynload::miopenSet4dTensorDescriptor(
        desc_, type, kernel_with_group[0], kernel_with_group[1], kernel_with_group[2], kernel_with_group[3]));
    return desc_;
  }

  template <typename T>
  inline miopenTensorDescriptor_t descriptor(const DataLayout& order,
                                            const std::vector<int>& kernel,
                                            const int groups = 1) {
    return descriptor(/*GetCudnnTensorFormat(order),*/ CudnnDataType<T>::type,
                      kernel, groups);
  }
 private:
  miopenTensorDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedFilterDescriptor);
};

class ScopedConvolutionDescriptor {
 public:
  ScopedConvolutionDescriptor() {
    PADDLE_ENFORCE(dynload::miopenCreateConvolutionDescriptor(&desc_));
  }
  ~ScopedConvolutionDescriptor() {
    PADDLE_ENFORCE(dynload::miopenDestroyConvolutionDescriptor(desc_));
  }

  inline miopenConvolutionDescriptor_t descriptor(
      miopenDataType_t type, const std::vector<int>& pads,
      const std::vector<int>& strides, const std::vector<int>& dilations) {
    PADDLE_ENFORCE_EQ(pads.size(), strides.size());
    PADDLE_ENFORCE_EQ(pads.size(), dilations.size());
    if (pads.size()!=2){
        PADDLE_THROW("miopen only supports 2D Convolution, dim=%d not allowed",pads.size());
    }

    PADDLE_ENFORCE(dynload::miopenInitConvolutionDescriptor(
        desc_, miopenConvolution, pads[0], pads[1], strides[0], strides[1],
	dilations[0], dilations[1]));
    return desc_;
  }

  template <typename T>
  inline miopenConvolutionDescriptor_t descriptor(
      const std::vector<int>& pads, const std::vector<int>& strides,
      const std::vector<int>& dilations) {
    return descriptor(CudnnDataType<T>::type, pads, strides, dilations);
  }

 private:
  miopenConvolutionDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedConvolutionDescriptor);
};

class ScopedPoolingDescriptor {
 public:
  ScopedPoolingDescriptor() {
    PADDLE_ENFORCE(dynload::miopenCreatePoolingDescriptor(&desc_));
  }
  ~ScopedPoolingDescriptor() {
    PADDLE_ENFORCE(dynload::miopenDestroyPoolingDescriptor(desc_));
  }
  inline miopenPoolingDescriptor_t descriptor(const PoolingMode& mode,
                                             const std::vector<int>& kernel,
                                             const std::vector<int>& pads,
                                             const std::vector<int>& strides) {
    PADDLE_ENFORCE_EQ(kernel.size(), pads.size());
    PADDLE_ENFORCE_EQ(kernel.size(), strides.size());
    if (kernel.size()!=2){
        PADDLE_THROW("miopen only supports 2D Pooling, dim=%d not allowed",kernel.size());
    }

    PADDLE_ENFORCE(dynload::miopenSet2dPoolingDescriptor(
        desc_, (mode == PoolingMode::kMaximum
                    ? miopenPoolingMax
                    : miopenPoolingAverage),
        kernel[0], kernel[1], pads[0], pads[1], strides[0], strides[1]));
    return desc_;
  }
 private:
  miopenPoolingDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedPoolingDescriptor);
};

}  // namespace platform
}  // namespace paddle
