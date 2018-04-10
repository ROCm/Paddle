/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "hl_cuda_cudnn.h"
#include <miopen/miopen.h>
#include <gflags/gflags.h>
#include "hl_cuda_cudnn.ph"
#include "hl_thread.ph"
#include "paddle/utils/DynamicLoader.h"
#include "paddle/utils/Logging.h"

DEFINE_int32(cudnn_conv_workspace_limit_in_mb,
             4096,
             "Specify cuDNN max workspace limit, in units MB, "
             "4096MB=4GB by default.");

namespace dynload {

std::once_flag cudnn_dso_flag;
void* cudnn_dso_handle = nullptr;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cudbnn routine
 * via operator overloading: operator ()
 *
 * note: default dynamic linked libs
 **/

#ifdef PADDLE_USE_DSO

#define DYNAMIC_LOAD_CUDNN_WRAP(__name)                                     \
  struct DynLoad__##__name {                                                \
    template <typename... Args>                                             \
    auto operator()(Args... args) -> decltype(__name(args...)) {            \
      using cudnn_func = decltype(__name(args...)) (*)(Args...);            \
      std::call_once(cudnn_dso_flag, GetCudnnDsoHandle, &cudnn_dso_handle); \
      void* p_##__name = dlsym(cudnn_dso_handle, #__name);                  \
      return reinterpret_cast<cudnn_func>(p_##__name)(args...);             \
    }                                                                       \
  } __name; /* struct DynLoad__##__name */

#else

#define DYNAMIC_LOAD_CUDNN_WRAP(__name)                          \
  struct DynLoad__##__name {                                     \
    template <typename... Args>                                  \
    auto operator()(Args... args) -> decltype(__name(args...)) { \
      return __name(args...);                                    \
    }                                                            \
  } __name; /* struct DynLoad__##__name */

#endif

/**
 * include all needed cudnn functions in HPPL
 * different cudnn version has different interfaces
 **/
// clang-format off
#define CUDNN_DNN_ROUTINE_EACH(__macro)                   \
  __macro(miopenSet4dTensorDescriptor)                     \
  __macro(miopenFindConvolutionForwardAlgorithm)             \
  __macro(miopenCreateTensorDescriptor)                    \
  __macro(miopenDestroyTensorDescriptor)                   \
  __macro(miopenSet2dPoolingDescriptor)                    \
  __macro(miopenCreateConvolutionDescriptor)               \
  __macro(miopenCreatePoolingDescriptor)                   \
  __macro(miopenDestroyPoolingDescriptor)                  \
  __macro(miopenInitConvolutionDescriptor)                \
  __macro(miopenDestroyConvolutionDescriptor)              \
  __macro(miopenCreate)                                    \
  __macro(miopenDestroy)                                   \
  __macro(miopenSetStream)                                 \
  __macro(miopenActivationForward)                         \
  __macro(miopenConvolutionForward)                        \
  __macro(miopenConvolutionBackwardBias)                   \
  __macro(miopenConvolutionForwardGetWorkSpaceSize)        \
  __macro(miopenTransformTensor)                           \
  __macro(miopenPoolingGetWorkSpaceSize)                   \
  __macro(miopenPoolingForward)                            \
  __macro(miopenPoolingBackward)                           \
  __macro(miopenSoftmaxBackward)                           \
  __macro(miopenSoftmaxForward)
CUDNN_DNN_ROUTINE_EACH(DYNAMIC_LOAD_CUDNN_WRAP)

#define CUDNN_DNN_ROUTINE_EACH_R2(__macro)                \
  __macro(miopenOpTensor)                                 \
  __macro(miopenConvolutionBackwardData)                   \
  __macro(miopenConvolutionBackwardWeights)
CUDNN_DNN_ROUTINE_EACH_R2(DYNAMIC_LOAD_CUDNN_WRAP)

// APIs available after R3:
#define CUDNN_DNN_ROUTINE_EACH_AFTER_R3(__macro)              \
  __macro(miopenConvolutionBackwardWeightsGetWorkSpaceSize)   \
  __macro(miopenFindConvolutionBackwardDataAlgorithm)           \
  __macro(miopenFindConvolutionBackwardWeightsAlgorithm)         \
  __macro(miopenConvolutionBackwardDataGetWorkSpaceSize)
CUDNN_DNN_ROUTINE_EACH_AFTER_R3(DYNAMIC_LOAD_CUDNN_WRAP)
#undef CUDNN_DNN_ROUTINE_EACH_AFTER_R3


// APIs available after R4:
#define CUDNN_DNN_ROUTINE_EACH_AFTER_R4(__macro)             \
  __macro(miopenBatchNormalizationForwardTraining)            \
  __macro(miopenBatchNormalizationForwardInference)           \
  __macro(miopenBatchNormalizationBackward)
CUDNN_DNN_ROUTINE_EACH_AFTER_R4(DYNAMIC_LOAD_CUDNN_WRAP)
#undef CUDNN_DNN_ROUTINE_EACH_AFTER_R4

// APIs in R5
#define CUDNN_DNN_ROUTINE_EACH_R5(__macro)                    \
  __macro(miopenCreateActivationDescriptor)                    \
  __macro(miopenSetActivationDescriptor)                       \
  __macro(miopenGetActivationDescriptor)                       \
  __macro(miopenDestroyActivationDescriptor)
CUDNN_DNN_ROUTINE_EACH_R5(DYNAMIC_LOAD_CUDNN_WRAP)
#undef CUDNN_DNN_ROUTINE_EACH_R5

#undef CUDNN_DNN_ROUTINE_EACH
// clang-format on
} /* namespace dynload */

/**
 * Check build-in cudnn function using glog and it **does not**
 * support << operator for more details error info.
 */
#define CHECK_CUDNN(cudnnFunc)                                         \
  do {                                                                 \
    miopenStatus_t cudnnStat = cudnnFunc;                               \
    CHECK_EQ(miopenStatusSuccess, cudnnStat)                          \
        << "MIOpen Error: "; \
  } while (0)

bool g_is_libcudnn_init = false;
int g_cudnn_lib_version = 0;

void hl_cudnn_desc_init(miopenTensorDescriptor_t* cudnn_desc) {
  CHECK_CUDNN(dynload::miopenCreateTensorDescriptor(cudnn_desc));
}

void hl_cudnn_init(miopenHandle_t* cudnn_handle, hipStream_t stream) {
  CHECK_CUDNN(dynload::miopenCreate(cudnn_handle));
  CHECK_CUDNN(dynload::miopenSetStream(*cudnn_handle, stream));

  g_is_libcudnn_init = true;
}

int hl_get_cudnn_lib_version() { return g_cudnn_lib_version; }

void hl_conv_workspace(hl_tensor_descriptor input,
                       hl_tensor_descriptor output,
                       hl_filter_descriptor filter,
                       hl_convolution_descriptor conv,
                       int* convFwdAlgo,
                       size_t* fwdLimitBytes,
                       int* convBwdDataAlgo,
                       size_t* bwdDataLimitBytes,
                       int* convBwdFilterAlgo,
                       size_t* bwdFilterLimitBytes,
                       bool useDilation) {

  CHECK_NOTNULL(input);
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(filter);
  CHECK_NOTNULL(conv);

  CHECK_NOTNULL(convFwdAlgo);
  CHECK_NOTNULL(convBwdDataAlgo);
  CHECK_NOTNULL(convBwdFilterAlgo);
#if 0
  // Specify workspace limit directly
  size_t memoryLimitBytes =
      (1LL << 20) * FLAGS_cudnn_conv_workspace_limit_in_mb;
#endif

  // For dilation
  int algo = 0;

  // cudnn convolution forward configuration
  miopenTensorDescriptor_t fwd_src_desc = GET_TENSOR_DESCRIPTOR(input);
  miopenTensorDescriptor_t fwd_dest_desc = GET_TENSOR_DESCRIPTOR(output);
  miopenTensorDescriptor_t fwd_filter_desc = GET_FILTER_DESCRIPTOR(filter);
  miopenConvolutionDescriptor_t fwd_conv_desc = GET_CONVOLUTION_DESCRIPTOR(conv);
  // cudnn convolution backward data configuration
  miopenTensorDescriptor_t bwd_data_filter_desc = GET_FILTER_DESCRIPTOR(filter);
  miopenTensorDescriptor_t bwd_data_diff_desc = GET_TENSOR_DESCRIPTOR(output);
  miopenTensorDescriptor_t bwd_data_grad_desc = GET_TENSOR_DESCRIPTOR(input);
  miopenConvolutionDescriptor_t bwd_data_conv_desc =
      GET_CONVOLUTION_DESCRIPTOR(conv);
  // cudnn convolution backward filter configuration
  miopenTensorDescriptor_t bwd_filter_src_desc = GET_TENSOR_DESCRIPTOR(input);
  miopenTensorDescriptor_t bwd_filter_diff_desc = GET_TENSOR_DESCRIPTOR(output);
  miopenConvolutionDescriptor_t bwd_filter_conv_desc =
      GET_CONVOLUTION_DESCRIPTOR(conv);
  miopenTensorDescriptor_t bwd_filter_grad_desc = GET_FILTER_DESCRIPTOR(filter);

  if (useDilation) {
    convFwdAlgo = &algo;
    convBwdDataAlgo = &algo;
    convBwdFilterAlgo = &algo;
  } else {
    convFwdAlgo = static_cast<int>(miopenConvolutionFwdAlgoGEMM);
    convBwdDataAlgo = static_cast<int>(miopenConvolutionBwdDataAlgoGEMM);
    convBwdFilterAlgo = static_cast<int>(miopenConvolutionBwdWeightsAlgoGEMM);
  }

  CHECK_CUDNN(dynload::miopenConvolutionForwardGetWorkSpaceSize(
      t_resource.cudnn_handle,
      fwd_src_desc,
      fwd_filter_desc,
      fwd_conv_desc,
      fwd_dest_desc,
      fwdLimitBytes));

  CHECK_CUDNN(dynload::miopenConvolutionBackwardDataGetWorkSpaceSize(
      t_resource.cudnn_handle,
      bwd_data_filter_desc,
      bwd_data_diff_desc,
      bwd_data_conv_desc,
      bwd_data_grad_desc,
      bwdDataLimitBytes));

  CHECK_CUDNN(dynload::miopenConvolutionBackwardWeightsGetWorkSpaceSize(
      t_resource.cudnn_handle,
      bwd_filter_src_desc,
      bwd_filter_diff_desc,
      bwd_filter_conv_desc,
      bwd_filter_grad_desc,
      bwdFilterLimitBytes));

}


void hl_create_tensor_descriptor(hl_tensor_descriptor* image_desc,
                                 int batch_size,
                                 int feature_maps,
                                 int height,
                                 int width) {
  CHECK_NOTNULL(image_desc);

  cudnn_tensor_descriptor hl_desc =
      (cudnn_tensor_descriptor)malloc(sizeof(_cudnn_tensor_descriptor));
  CHECK_NOTNULL(hl_desc);

#ifndef PADDLE_TYPE_DOUBLE
  miopenDataType_t data_type = miopenFloat;
#else
  //cudnnDataType_t data_type = CUDNN_DATA_DOUBLE;
#endif
  CHECK_CUDNN(dynload::miopenCreateTensorDescriptor(&hl_desc->desc));

  CHECK_CUDNN(dynload::miopenSet4dTensorDescriptor(hl_desc->desc,
                                                  data_type,
                                                  batch_size,
                                                  feature_maps,
                                                  height,
                                                  width));

  //hl_desc->format = CUDNN_TENSOR_NCHW;
  hl_desc->data_type = data_type;
  hl_desc->batch_size = batch_size;
  hl_desc->feature_maps = feature_maps;
  hl_desc->height = height;
  hl_desc->width = width;

  *image_desc = (hl_tensor_descriptor)hl_desc;
}

void hl_create_tensor_descriptor(hl_tensor_descriptor* image_desc) {
  CHECK_NOTNULL(image_desc);

  cudnn_tensor_descriptor hl_desc =
      (cudnn_tensor_descriptor)malloc(sizeof(_cudnn_tensor_descriptor));
  CHECK_NOTNULL(hl_desc);

#ifndef PADDLE_TYPE_DOUBLE
  miopenDataType_t data_type = miopenFloat;
#else
  //cudnnDataType_t data_type = CUDNN_DATA_DOUBLE;
#endif
  CHECK_CUDNN(dynload::miopenCreateTensorDescriptor(&hl_desc->desc));

  hl_desc->data_type = data_type;

  *image_desc = (hl_tensor_descriptor)hl_desc;
}

void hl_tensor_reshape(hl_tensor_descriptor image_desc,
                       int batch_size,
                       int feature_maps,
                       int height,
                       int width) {
  const int stride_w = 1;
  const int stride_h = width * stride_w;
  const int stride_c = height * stride_h;
  const int stride_n = feature_maps * stride_c;
  return hl_tensor_reshape(image_desc,
                           batch_size,
                           feature_maps,
                           height,
                           width,
                           stride_n,
                           stride_c,
                           stride_h,
                           stride_w);
}

void hl_tensor_reshape(hl_tensor_descriptor image_desc,
                       int batch_size,
                       int feature_maps,
                       int height,
                       int width,
                       int nStride,
                       int cStride,
                       int hStride,
                       int wStride) {
  CHECK_NOTNULL(image_desc);

  cudnn_tensor_descriptor hl_desc = (cudnn_tensor_descriptor)image_desc;
  CHECK_NOTNULL(hl_desc->desc);
  if ((wStride != 1) || (hStride != width * wStride) ||
      (cStride != height * hStride) || (nStride != feature_maps * cStride)) {
    LOG(FATAL) << "Invalid Stride for TensorDescriptor ";
  }

  CHECK_CUDNN(dynload::miopenSet4dTensorDescriptor(hl_desc->desc,
                                                   hl_desc->data_type,
                                                   batch_size,
                                                   feature_maps,
                                                   height,
                                                   width));

  hl_desc->batch_size = batch_size;
  hl_desc->feature_maps = feature_maps;
  hl_desc->height = height;
  hl_desc->width = width;
}

void hl_destroy_tensor_descriptor(hl_tensor_descriptor image_desc) {
  CHECK_NOTNULL(image_desc);

  cudnn_tensor_descriptor hl_desc = (cudnn_tensor_descriptor)image_desc;
  CHECK_NOTNULL(hl_desc->desc);

  CHECK_CUDNN(dynload::miopenDestroyTensorDescriptor(hl_desc->desc));

  hl_desc->desc = NULL;

  free(image_desc);
}

void hl_create_pooling_descriptor(hl_pooling_descriptor* pooling_desc,
                                  hl_pooling_mode_t mode,
                                  int height,
                                  int width,
                                  int height_padding,
                                  int width_padding,
                                  int stride_height,
                                  int stride_width) {
  miopenPoolingMode_t cudnn_mode;
  switch (mode) {
    case HL_POOLING_MAX:
      cudnn_mode = miopenPoolingMax;
      break;
    case HL_POOLING_AVERAGE:
      cudnn_mode = miopenPoolingAverage;
      break;
    case HL_POOLING_AVERAGE_INCLUDE_PADDING:
      cudnn_mode = miopenPoolingAverage;
      LOG(FATAL) << "parameter mode error";
      break;
    default:
      LOG(FATAL) << "parameter mode error";
  }

  CHECK_NOTNULL(pooling_desc);

  cudnn_pooling_descriptor hl_pooling_desc =
      (cudnn_pooling_descriptor)malloc(sizeof(_cudnn_pooling_descriptor));
  CHECK_NOTNULL(hl_pooling_desc);

  CHECK_CUDNN(dynload::miopenCreatePoolingDescriptor(&hl_pooling_desc->desc));

  CHECK_CUDNN(dynload::miopenSet2dPoolingDescriptor(hl_pooling_desc->desc,
                                                   cudnn_mode,
                                                   height,
                                                   width,
                                                   height_padding,
                                                   width_padding,
                                                   stride_height,
                                                   stride_width));

  hl_pooling_desc->mode = cudnn_mode;
  hl_pooling_desc->window_height = height;
  hl_pooling_desc->window_width = width;
  hl_pooling_desc->stride_height = stride_height;
  hl_pooling_desc->stride_width = stride_width;

  *pooling_desc = (hl_pooling_descriptor)hl_pooling_desc;
}

void hl_destroy_pooling_descriptor(hl_pooling_descriptor pooling_desc) {
  CHECK_NOTNULL(pooling_desc);

  cudnn_pooling_descriptor hl_pooling = (cudnn_pooling_descriptor)pooling_desc;

  CHECK_NOTNULL(hl_pooling->desc);
  CHECK_CUDNN(dynload::miopenDestroyPoolingDescriptor(hl_pooling->desc));

  hl_pooling->desc = NULL;

  free(pooling_desc);
}

void hl_pooling_workspace(hl_tensor_descriptor output,
			  size_t* sizeInBytes) {
    CHECK_NOTNULL(sizeInBytes);
    CHECK_NOTNULL(output);
    miopenTensorDescriptor_t output_desc = ((cudnn_tensor_descriptor)output)->desc;
    CHECK_CUDNN(dynload::miopenPoolingGetWorkSpaceSize(output_desc, sizeInBytes));
}

void hl_pooling_forward(hl_tensor_descriptor input,
                        real* input_image,
                        hl_tensor_descriptor output,
                        real* output_image,
                        hl_pooling_descriptor pooling) {
  miopenPoolingDescriptor_t pooling_desc;
  miopenTensorDescriptor_t input_desc;
  miopenTensorDescriptor_t output_desc;

  CHECK_NOTNULL(input);
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(pooling);
  CHECK_NOTNULL(input_image);
  CHECK_NOTNULL(output_image);

  real alpha = 1.0f;
  real beta = 1.0f;
  input_desc = ((cudnn_tensor_descriptor)input)->desc;
  output_desc = ((cudnn_tensor_descriptor)output)->desc;
  pooling_desc = ((cudnn_pooling_descriptor)pooling)->desc;
  size_t sizeInBytes = 0;
  CHECK_CUDNN(dynload::miopenPoolingGetWorkSpaceSize(output_desc, &sizeInBytes));
  void* gpuWorkSpace = hl_malloc_device(sizeInBytes);
  CHECK_NOTNULL(gpuWorkSpace);
  CHECK_CUDNN(dynload::miopenPoolingForward(t_resource.cudnn_handle,
                                           pooling_desc,
                                           &alpha,
                                           input_desc,
                                           input_image,
                                           &beta,
                                           output_desc,
                                           output_image,
                                           false,
                                           gpuWorkSpace,
                                           sizeInBytes));
  CHECK_SYNC("hl_pooling_forward failed");
  hl_free_mem_device(gpuWorkSpace);
}

void hl_pooling_backward(hl_tensor_descriptor input,
                         real* input_image,
                         real* input_image_grad,
                         hl_tensor_descriptor output,
                         real* output_image,
                         real* output_image_grad,
                         hl_pooling_descriptor pooling) {
  miopenPoolingDescriptor_t pooling_desc;
  miopenTensorDescriptor_t input_desc;
  miopenTensorDescriptor_t output_desc;

  CHECK_NOTNULL(input);
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(pooling);
  CHECK_NOTNULL(input_image);
  CHECK_NOTNULL(input_image_grad);
  CHECK_NOTNULL(output_image);
  CHECK_NOTNULL(output_image_grad);

  real alpha = 1.0f;
  real beta = 1.0f;
  input_desc = ((cudnn_tensor_descriptor)input)->desc;
  output_desc = ((cudnn_tensor_descriptor)output)->desc;
  pooling_desc = ((cudnn_pooling_descriptor)pooling)->desc;
  size_t sizeInBytes = 0;
  CHECK_CUDNN(dynload::miopenPoolingGetWorkSpaceSize(output_desc, &sizeInBytes));
  void* gpuWorkSpace = hl_malloc_device(sizeInBytes);
  CHECK_NOTNULL(gpuWorkSpace);
  CHECK_CUDNN(dynload::miopenPoolingBackward(t_resource.cudnn_handle,
                                            pooling_desc,
                                            &alpha,
                                            output_desc,
                                            output_image,
                                            output_desc,
                                            output_image_grad,
                                            input_desc,
                                            input_image,
                                            &beta,
                                            input_desc,
                                            input_image_grad,
                                            gpuWorkSpace));
  CHECK_SYNC("hl_pooling_backward failed");
  hl_free_mem_device(gpuWorkSpace);
}

void hl_create_filter_descriptor(hl_filter_descriptor* filter,
                                 int input_feature_maps,
                                 int output_feature_maps,
                                 int height,
                                 int width) {
  CHECK_NOTNULL(filter);

  cudnn_filter_descriptor hl_filter =
      (cudnn_filter_descriptor)malloc(sizeof(_cudnn_filter_descriptor));
  CHECK_NOTNULL(hl_filter);

  CHECK_CUDNN(dynload::miopenCreateTensorDescriptor(&hl_filter->desc));

#ifndef PADDLE_TYPE_DOUBLE
  miopenDataType_t data_type = miopenFloat;
#else
  //cudnnDataType_t data_type = CUDNN_DATA_DOUBLE;
#endif
  CHECK_CUDNN(dynload::miopenSet4dTensorDescriptor(hl_filter->desc,
                                                  data_type,
                                                  output_feature_maps,
                                                  input_feature_maps,
                                                  height,
                                                  width));

  hl_filter->data_type = data_type;
  hl_filter->output_feature_maps = output_feature_maps;
  hl_filter->input_feature_maps = input_feature_maps;
  hl_filter->filter_height = height;
  hl_filter->filter_width = width;

  *filter = (hl_filter_descriptor)hl_filter;
}

void hl_destroy_filter_descriptor(hl_filter_descriptor filter) {
  CHECK_NOTNULL(filter);

  cudnn_filter_descriptor hl_filter = (cudnn_filter_descriptor)filter;
  CHECK_NOTNULL(hl_filter->desc);

  CHECK_CUDNN(dynload::miopenDestroyTensorDescriptor(hl_filter->desc));

  hl_filter->desc = NULL;

  free(filter);
}

void hl_create_convolution_descriptor(hl_convolution_descriptor* conv,
                                      hl_tensor_descriptor image,
                                      hl_filter_descriptor filter,
                                      int padding_height,
                                      int padding_width,
                                      int stride_height,
                                      int stride_width,
                                      int dilation_h,
                                      int dilation_w) {
  CHECK_NOTNULL(conv);

  cudnn_convolution_descriptor hl_conv = (cudnn_convolution_descriptor)malloc(
      sizeof(_cudnn_convolution_descriptor));

  CHECK_NOTNULL(hl_conv);
  CHECK_CUDNN(dynload::miopenCreateConvolutionDescriptor(&hl_conv->desc));

  miopenConvolutionMode_t mode = miopenConvolution;//CUDNN_CROSS_CORRELATION;

  if (dilation_h > 1 || dilation_w > 1) {
  }

  CHECK_CUDNN(dynload::miopenInitConvolutionDescriptor(hl_conv->desc,
                                                       mode,
                                                       padding_height,
                                                       padding_width,
                                                       stride_height,
                                                       stride_width,
                                                       dilation_h,
                                                       dilation_w));

  hl_conv->input_image = image;
  hl_conv->filter = filter;
  hl_conv->padding_height = padding_height;
  hl_conv->padding_width = padding_width;
  hl_conv->stride_height = stride_height;
  hl_conv->stride_width = stride_width;
  hl_conv->upscalex = 1;
  hl_conv->upscaley = 1;
  hl_conv->mode = mode;

  *conv = (hl_convolution_descriptor)hl_conv;
}

void hl_reset_convolution_descriptor(hl_convolution_descriptor conv,
                                     hl_tensor_descriptor image,
                                     hl_filter_descriptor filter,
                                     int padding_height,
                                     int padding_width,
                                     int stride_height,
                                     int stride_width,
                                     int dilation_h,
                                     int dilation_w) {
  CHECK_NOTNULL(conv);
  CHECK_NOTNULL(image);
  CHECK_NOTNULL(filter);

  miopenConvolutionDescriptor_t conv_desc = GET_CONVOLUTION_DESCRIPTOR(conv);
  miopenConvolutionMode_t mode = miopenConvolution;//CUDNN_CROSS_CORRELATION;

  CHECK_CUDNN(dynload::miopenInitConvolutionDescriptor(conv_desc,
                                                       mode,
                                                       padding_height,
                                                       padding_width,
                                                       stride_height,
                                                       stride_width,
                                                       dilation_h,
                                                       dilation_w));

  cudnn_convolution_descriptor hl_conv = (cudnn_convolution_descriptor)conv;
  hl_conv->input_image = image;
  hl_conv->filter = filter;
  hl_conv->padding_height = padding_height;
  hl_conv->padding_width = padding_width;
  hl_conv->stride_height = stride_height;
  hl_conv->stride_width = stride_width;
  hl_conv->upscalex = 1;
  hl_conv->upscaley = 1;
  hl_conv->mode = mode;
}

void hl_destroy_convolution_descriptor(hl_convolution_descriptor conv) {
  CHECK_NOTNULL(conv);

  cudnn_convolution_descriptor hl_conv = (cudnn_convolution_descriptor)conv;
  CHECK_NOTNULL(hl_conv->desc);

  CHECK_CUDNN(dynload::miopenDestroyConvolutionDescriptor(hl_conv->desc));
  hl_conv->desc = NULL;

  free(conv);
}

void hl_convolution_forward(hl_tensor_descriptor input,
                            real* input_data,
                            hl_tensor_descriptor output,
                            real* output_data,
                            hl_filter_descriptor filter,
                            real* filter_data,
                            hl_convolution_descriptor conv,
                            void* gpuWorkSpace,
                            size_t sizeInBytes,
                            int convFwdAlgo) {
  CHECK_NOTNULL(input);
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(filter);
  CHECK_NOTNULL(conv);
  CHECK_NOTNULL(input_data);
  CHECK_NOTNULL(output_data);
  CHECK_NOTNULL(filter_data);
  miopenTensorDescriptor_t src_desc = GET_TENSOR_DESCRIPTOR(input);
  miopenTensorDescriptor_t dest_desc = GET_TENSOR_DESCRIPTOR(output);
  miopenTensorDescriptor_t filter_desc = GET_FILTER_DESCRIPTOR(filter);
  miopenConvolutionDescriptor_t conv_desc = GET_CONVOLUTION_DESCRIPTOR(conv);
  real alpha = 1.0f;
  real beta = 1.0f;
  int algoCount = 0;
  miopenConvAlgoPerf_t perfRes;
  CHECK_CUDNN(dynload::miopenFindConvolutionForwardAlgorithm(
      t_resource.cudnn_handle,
      src_desc,
      input_data,
      filter_desc,
      filter_data,
      conv_desc,
      dest_desc,
      output_data,
      1,
      &algoCount,
      &perfRes,
      gpuWorkSpace,
      sizeInBytes,
      false));

  CHECK_CUDNN(dynload::miopenConvolutionForward(
      t_resource.cudnn_handle,
      &alpha,
      src_desc,
      input_data,
      filter_desc,
      filter_data,
      conv_desc,
      perfRes.fwd_algo,
      &beta,
      dest_desc,
      output_data,
      gpuWorkSpace,
      sizeInBytes));
  CHECK_SYNC("hl_convolution_forward failed");
}

void hl_convolution_forward_add_bias(hl_tensor_descriptor bias,
                                     real* bias_data,
                                     hl_tensor_descriptor output,
                                     real* output_data) {
  CHECK_NOTNULL(bias);
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(bias_data);
  CHECK_NOTNULL(output_data);

  miopenTensorDescriptor_t output_desc = GET_TENSOR_DESCRIPTOR(output);
  miopenTensorDescriptor_t bias_desc = GET_TENSOR_DESCRIPTOR(bias);
  real alpha = 1.0f;
  real alpha1= 0.0f;
  real beta = 1.0f;

  CHECK_CUDNN(dynload::miopenOpTensor(t_resource.cudnn_handle,
                                      miopenTensorOp_t::miopenTensorOpAdd,
                                      &alpha,
                                      bias_desc,
                                      bias_data,
                                      &alpha1,
                                      bias_desc,
                                      bias_data,
                                      &beta,
                                      output_desc,
                                      output_data));
  CHECK_SYNC("hl_convolution_forward_add_bias failed");
}

void hl_convolution_backward_bias(hl_tensor_descriptor bias,
                                  real* bias_grad_data,
                                  hl_tensor_descriptor output,
                                  real* output_grad_data) {
  CHECK_NOTNULL(bias);
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(bias_grad_data);
  CHECK_NOTNULL(output_grad_data);

  real alpha = 1.0f;
  real beta = 1.0f;
  miopenTensorDescriptor_t diff_desc = GET_TENSOR_DESCRIPTOR(output);
  miopenTensorDescriptor_t bias_desc = GET_TENSOR_DESCRIPTOR(bias);
  CHECK_CUDNN(dynload::miopenConvolutionBackwardBias(t_resource.cudnn_handle,
                                                    &alpha,
                                                    diff_desc,
                                                    output_grad_data,
                                                    &beta,
                                                    bias_desc,
                                                    bias_grad_data));
  CHECK_SYNC("hl_convolution_backward_bias failed");
}

void hl_convolution_backward_filter(hl_tensor_descriptor input,
                                    real* input_data,
                                    hl_tensor_descriptor output,
                                    real* output_grad_data,
                                    hl_filter_descriptor filter,
                                    real* filter_grad_data,
                                    hl_convolution_descriptor conv,
                                    void* gpuWorkSpace,
                                    size_t sizeInBytes,
                                    int convBwdFilterAlgo) {
  CHECK_NOTNULL(input);
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(filter);
  CHECK_NOTNULL(conv);
  CHECK_NOTNULL(input_data);
  CHECK_NOTNULL(output_grad_data);
  CHECK_NOTNULL(filter_grad_data);

  real alpha = 1.0f;
  real beta = 1.0f;
  int algoCount = 0;
  miopenConvAlgoPerf_t perfRes;
  miopenTensorDescriptor_t src_desc = GET_TENSOR_DESCRIPTOR(input);
  miopenTensorDescriptor_t diff_desc = GET_TENSOR_DESCRIPTOR(output);
  miopenConvolutionDescriptor_t conv_desc = GET_CONVOLUTION_DESCRIPTOR(conv);
  miopenTensorDescriptor_t grad_desc = GET_FILTER_DESCRIPTOR(filter);

  CHECK_CUDNN(dynload::miopenFindConvolutionBackwardWeightsAlgorithm(
      t_resource.cudnn_handle,
      src_desc,
      input_data, //?
      diff_desc,
      output_grad_data,
      conv_desc,
      grad_desc,
      filter_grad_data,
      1,
      &algoCount,
      &perfRes,
      gpuWorkSpace,
      sizeInBytes,
      false));

  CHECK_CUDNN(dynload::miopenConvolutionBackwardWeights(
      t_resource.cudnn_handle,
      &alpha,
      diff_desc,
      output_grad_data,
      src_desc,
      input_data,
      conv_desc,
      perfRes.bwd_weights_algo,
      &beta,
      grad_desc,
      filter_grad_data,
      gpuWorkSpace,
      sizeInBytes));
  CHECK_SYNC("hl_convolution_backward_filter failed");
}

void hl_convolution_backward_data(hl_tensor_descriptor input,
                                  real* input_data_grad,
                                  hl_tensor_descriptor output,
                                  real* output_grad_data,
                                  hl_filter_descriptor filter,
                                  real* filter_data,
                                  hl_convolution_descriptor conv,
                                  void* gpuWorkSpace,
                                  size_t sizeInBytes,
                                  int convBwdDataAlgo) {
  real alpha = 1.0f;
  real beta = 1.0f;
  int algoCount = 0;
  miopenConvAlgoPerf_t perfRes;
  miopenTensorDescriptor_t filter_desc = GET_FILTER_DESCRIPTOR(filter);
  miopenTensorDescriptor_t diff_desc = GET_TENSOR_DESCRIPTOR(output);
  miopenTensorDescriptor_t grad_desc = GET_TENSOR_DESCRIPTOR(input);
  miopenConvolutionDescriptor_t conv_desc = GET_CONVOLUTION_DESCRIPTOR(conv);

  CHECK_CUDNN(dynload::miopenFindConvolutionBackwardDataAlgorithm(
      t_resource.cudnn_handle,
      filter_desc,
      filter_data,
      diff_desc,
      output_grad_data,
      conv_desc,
      grad_desc,
      input_data_grad,
      1,
      &algoCount,
      &perfRes,
      gpuWorkSpace,
      sizeInBytes,
      false));

  CHECK_CUDNN(dynload::miopenConvolutionBackwardData(
      t_resource.cudnn_handle,
      &alpha,
      diff_desc,
      output_grad_data,
      filter_desc,
      filter_data,
      conv_desc,
      perfRes.bwd_data_algo,
      &beta,
      grad_desc,
      input_data_grad,
      gpuWorkSpace,
      sizeInBytes));
  CHECK_SYNC("hl_convolution_backward_data failed");
}

void hl_softmax_forward(real* input, real* output, int height, int width) {
#ifndef PADDLE_TYPE_DOUBLE
  miopenDataType_t data_type = miopenFloat;
#else
  //cudnnDataType_t data_type = CUDNN_DATA_DOUBLE;
#endif
  CHECK_CUDNN(dynload::miopenSet4dTensorDescriptor(t_resource.cudnn_desc,
                                                  data_type,
                                                  height,
                                                  width,
                                                  1,
                                                  1));

  real alpha = 1.0f;
  real beta = 0.0f;
  CHECK_CUDNN(dynload::miopenSoftmaxForward(t_resource.cudnn_handle,
                                           &alpha,
                                           t_resource.cudnn_desc,
                                           input,
                                           &beta,
                                           t_resource.cudnn_desc,
                                           output));
  CHECK_SYNC("hl_softmax_forward failed");
}

void hl_softmax_backward(real* output_value,
                         real* output_grad,
                         int height,
                         int width) {
#ifndef PADDLE_TYPE_DOUBLE
  miopenDataType_t data_type = miopenFloat;
#else
  //cudnnDataType_t data_type = CUDNN_DATA_DOUBLE;
#endif
  CHECK_CUDNN(dynload::miopenSet4dTensorDescriptor(t_resource.cudnn_desc,
                                                  data_type,
                                                  height,
                                                  width,
                                                  1,
                                                  1));

  real alpha = 1.0f;
  real beta = 0.0f;
  CHECK_CUDNN(dynload::miopenSoftmaxBackward(t_resource.cudnn_handle,
                                            &alpha,
                                            t_resource.cudnn_desc,
                                            output_value,
                                            t_resource.cudnn_desc,
                                            output_grad,
                                            &beta,
                                            t_resource.cudnn_desc,
                                            output_grad));
  CHECK_SYNC("hl_softmax_backward failed");
}

void hl_batch_norm_forward_training(hl_tensor_descriptor inputDesc,
                                    real* input,
                                    hl_tensor_descriptor outputDesc,
                                    real* output,
                                    hl_tensor_descriptor bnParamDesc,
                                    real* scale,
                                    real* bias,
                                    double factor,
                                    real* runningMean,
                                    real* runningInvVar,
                                    double epsilon,
                                    real* savedMean,
                                    real* savedVar) {
  if ((NULL != runningMean && NULL == runningInvVar) ||
      (NULL == runningMean && NULL != runningInvVar)) {
    LOG(FATAL) << "runningMean and runningInvVar can be NULL "
               << "but only at the same time.";
  }
  if ((NULL != savedMean && NULL == savedVar) ||
      (NULL == savedMean && NULL != savedVar)) {
    LOG(FATAL) << "savedMean and savedVar can be NULL "
               << "but only at the same time.";
  }

  miopenTensorDescriptor_t xDesc = GET_TENSOR_DESCRIPTOR(inputDesc);
  miopenTensorDescriptor_t yDesc = GET_TENSOR_DESCRIPTOR(outputDesc);
  miopenTensorDescriptor_t bnDesc = GET_TENSOR_DESCRIPTOR(bnParamDesc);
  real alpha = 1.0f;
  real beta = 1.0f;
  miopenBatchNormMode_t mode = miopenBNSpatial;
  CHECK_CUDNN(
      dynload::miopenBatchNormalizationForwardTraining(t_resource.cudnn_handle,
                                                      mode,
                                                      &alpha,
                                                      &beta,
                                                      xDesc,
                                                      input,
                                                      yDesc,
                                                      output,
                                                      bnDesc,
                                                      scale,
                                                      bias,
                                                      factor,
                                                      runningMean,
                                                      runningInvVar,
                                                      epsilon,
                                                      savedMean,
                                                      savedVar));

  CHECK_SYNC("hl_batch_norm_forward_training failed");
}

void hl_batch_norm_forward_inference(hl_tensor_descriptor inputDesc,
                                     real* input,
                                     hl_tensor_descriptor outputDesc,
                                     real* output,
                                     hl_tensor_descriptor bnParamDesc,
                                     real* scale,
                                     real* bias,
                                     real* estimatedMean,
                                     real* estimatedInvVar,
                                     double epsilon) {
  miopenTensorDescriptor_t xDesc = GET_TENSOR_DESCRIPTOR(inputDesc);
  miopenTensorDescriptor_t yDesc = GET_TENSOR_DESCRIPTOR(outputDesc);
  miopenTensorDescriptor_t bnDesc = GET_TENSOR_DESCRIPTOR(bnParamDesc);
  real alpha = 1.0f;
  real beta = 1.0f;
  miopenBatchNormMode_t mode = miopenBNSpatial;

  CHECK_CUDNN(
      dynload::miopenBatchNormalizationForwardInference(t_resource.cudnn_handle,
                                                       mode,
                                                       &alpha,
                                                       &beta,
                                                       xDesc,
                                                       input,
                                                       yDesc,
                                                       output,
                                                       bnDesc,
                                                       scale,
                                                       bias,
                                                       estimatedMean,
                                                       estimatedInvVar,
                                                       epsilon));

  CHECK_SYNC("hl_batch_norm_forward_inference failed");
}

void hl_batch_norm_backward(hl_tensor_descriptor inputDesc,
                            real* input,
                            hl_tensor_descriptor outGradDesc,
                            real* outGrad,
                            hl_tensor_descriptor inGradDesc,
                            real* inGrad,
                            hl_tensor_descriptor dBnParamDesc,
                            real* scale,
                            real* scaleGrad,
                            real* biasGrad,
                            double epsilon,
                            real* savedMean,
                            real* savedInvVar) {
  if ((NULL != savedMean && NULL == savedInvVar) ||
      (NULL == savedMean && NULL != savedInvVar)) {
    LOG(FATAL) << "savedMean and savedVar can be NULL "
               << "but only at the same time.";
  }

  miopenTensorDescriptor_t xDesc = GET_TENSOR_DESCRIPTOR(inputDesc);
  miopenTensorDescriptor_t dyDesc = GET_TENSOR_DESCRIPTOR(outGradDesc);
  miopenTensorDescriptor_t dxDesc = GET_TENSOR_DESCRIPTOR(inGradDesc);
  miopenTensorDescriptor_t bnDesc = GET_TENSOR_DESCRIPTOR(dBnParamDesc);
  real alpha = 1.0f;
  real beta = 1.0f;
  miopenBatchNormMode_t mode = miopenBNSpatial;
  CHECK_CUDNN(dynload::miopenBatchNormalizationBackward(t_resource.cudnn_handle,
                                                       mode,
                                                       &alpha,
                                                       &beta,
                                                       &alpha,
                                                       &beta,
                                                       xDesc,
                                                       input,
                                                       dyDesc,
                                                       outGrad,
                                                       dxDesc,
                                                       inGrad,
                                                       bnDesc,
                                                       scale,
                                                       scaleGrad,
                                                       biasGrad,
                                                       epsilon,
                                                       savedMean,
                                                       savedInvVar));

  CHECK_SYNC("hl_batch_norm_backward failed");
}
