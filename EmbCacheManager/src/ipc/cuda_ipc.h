#pragma once

#include <cuda_runtime.h>
#include <torch/script.h>
#include "../utils.h"

namespace distemb {

typedef std::vector<int64_t> PyIpcHandle;

PyIpcHandle get_cuda_ipc_handle(torch::Tensor input) {
  void *data = const_cast<void *>(input.storage().data());
  std::vector<int64_t> py_handle(
      AlignUp(sizeof(cudaIpcMemHandle_t), sizeof(int64_t)));
  cudaIpcMemHandle_t *handle = (cudaIpcMemHandle_t *)py_handle.data();
  CUDA_CALL(cudaIpcGetMemHandle(handle, data));

  return py_handle;
}

torch::Tensor open_cuda_ipc_tensor(PyIpcHandle py_handle, std::string dtype,
                                   std::vector<int64_t> shape) {
  void *ptr;
  cudaIpcMemHandle_t *handle = (cudaIpcMemHandle_t *)py_handle.data();
  CUDA_CALL(
      cudaIpcOpenMemHandle(&ptr, *handle, cudaIpcMemLazyEnablePeerAccess));
  torch::ScalarType type = string2dtype(dtype);
  torch::Tensor out = torch::from_blob(
      ptr, shape, torch::TensorOptions().dtype(type).device(torch::kCUDA));
  return out;
}

void close_cuda_ipc_tensor(torch::Tensor tensor) {
  void *ptr = const_cast<void *>(tensor.storage().data());
  CUDA_CALL(cudaIpcCloseMemHandle(ptr));
}

}  // namespace distemb