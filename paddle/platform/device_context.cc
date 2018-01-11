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

#include "paddle/platform/device_context.h"
#include "paddle/memory/memory.h"

namespace paddle {
namespace platform {

template <>
Eigen::DefaultDevice* DeviceContext::GetEigenDevice<
    platform::CPUPlace, Eigen::DefaultDevice>() const {
  return reinterpret_cast<const CPUDeviceContext*>(this)->eigen_device();
}

CPUDeviceContext::CPUDeviceContext() {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

CPUDeviceContext::CPUDeviceContext(CPUPlace place) {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

Eigen::DefaultDevice* CPUDeviceContext::eigen_device() const {
  return eigen_device_.get();
}

Place CPUDeviceContext::GetPlace() const { return CPUPlace(); }

#ifdef PADDLE_WITH_CUDA

template <>
Eigen::GpuDevice*
DeviceContext::GetEigenDevice<platform::GPUPlace, Eigen::GpuDevice>() const {
  return reinterpret_cast<const CUDADeviceContext*>(this)->eigen_device();
}

class EigenCudaStreamDevice : public Eigen::StreamInterface {
 public:
  EigenCudaStreamDevice() : scratch_(nullptr), semaphore_(nullptr) {
    Eigen::initializeDeviceProp();
  }
  ~EigenCudaStreamDevice() override {}

  void Reinitialize(const hipStream_t* cuda_stream, GPUPlace place) {
    stream_ = cuda_stream;
    place_ = place;
    device_prop_ = &Eigen::m_deviceProperties[place.device];
  }

  const hipStream_t& stream() const override { return *stream_; }

  const hipDeviceProp_t& deviceProperties() const override {
    return *device_prop_;
  }

  void* allocate(size_t num_bytes) const override {
    return paddle::memory::Alloc(place_, num_bytes);
  }

  void deallocate(void* buffer) const override {
    paddle::memory::Free(place_, buffer);
  }

  void* scratchpad() const override {
    if (scratch_ == NULL) {
      scratch_ = allocate(Eigen::kHipScratchSize + sizeof(unsigned int));
    }
    return scratch_;
  }

  unsigned int* semaphore() const override {
    if (semaphore_ == NULL) {
      char* scratch =
          static_cast<char*>(scratchpad()) + Eigen::kHipScratchSize;
      semaphore_ = reinterpret_cast<unsigned int*>(scratch);
      PADDLE_ENFORCE(
          hipMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_));
    }
    return semaphore_;
  }

 private:
  GPUPlace place_;
  const hipStream_t* stream_;         // not owned;
  const hipDeviceProp_t* device_prop_;  // not owned;
  mutable void* scratch_;
  mutable unsigned int* semaphore_;
};

CUDADeviceContext::CUDADeviceContext(GPUPlace place) : place_(place) {
  SetDeviceId(place_.device);
  PADDLE_ENFORCE(hipStreamCreate(&stream_));
  eigen_stream_.reset(new EigenCudaStreamDevice());
  eigen_stream_->Reinitialize(&stream_, place);
  eigen_device_.reset(new Eigen::GpuDevice(eigen_stream_.get()));
  PADDLE_ENFORCE(dynload::hipblasCreate(&cublas_handle_));
  PADDLE_ENFORCE(dynload::hipblasSetStream(cublas_handle_, stream_));
  PADDLE_ENFORCE(dynload::cudnnCreate(&cudnn_handle_));
  PADDLE_ENFORCE(dynload::cudnnSetStream(cudnn_handle_, stream_));
}

CUDADeviceContext::~CUDADeviceContext() {
  SetDeviceId(place_.device);
  Wait();
  PADDLE_ENFORCE(dynload::hipblasDestroy(cublas_handle_));
  PADDLE_ENFORCE(dynload::cudnnDestroy(cudnn_handle_));
  eigen_stream_.reset();
  eigen_device_.reset();
  PADDLE_ENFORCE(hipStreamDestroy(stream_));
}

Place CUDADeviceContext::GetPlace() const { return place_; }

void CUDADeviceContext::Wait() const {
  PADDLE_ENFORCE(hipStreamSynchronize(stream_));
  PADDLE_ENFORCE(hipGetLastError());
}

Eigen::GpuDevice* CUDADeviceContext::eigen_device() const {
  return eigen_device_.get();
}

hipblasHandle_t CUDADeviceContext::cublas_handle() const {
  return cublas_handle_;
}

//cudnnHandle_t CUDADeviceContext::cudnn_handle() const { return cudnn_handle_; }

hipStream_t CUDADeviceContext::stream() const { return stream_; }

#endif

}  // namespace platform
}  // namespace paddle
