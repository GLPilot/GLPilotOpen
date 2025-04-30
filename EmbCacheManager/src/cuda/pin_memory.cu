#include <cuda_runtime.h>
#include <torch/script.h>
#include "../utils.h"
#include "ops.h"

namespace distemb {
namespace cuda {

void PinTensor(torch::Tensor tensor) {
  CUDA_CALL(cudaHostRegister(const_cast<void *>(tensor.storage().data()),
                             tensor.numel() * tensor.element_size(),
                             cudaHostRegisterDefault));
}

void UnpinTensor(torch::Tensor tensor) {
  CUDA_CALL(cudaHostUnregister(const_cast<void *>(tensor.storage().data())));
}

void RegisterPtr(int64_t ptr, int64_t size) {
  CUDA_CALL(cudaHostRegister(reinterpret_cast<void *>(ptr), size,
                             cudaHostRegisterDefault));
}

void UnregisterPtr(int64_t ptr) {
  CUDA_CALL(cudaHostUnregister(reinterpret_cast<void *>(ptr)));
}

}  // namespace cuda
}  // namespace distemb