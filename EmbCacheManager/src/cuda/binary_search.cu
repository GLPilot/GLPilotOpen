#include <cuda_runtime.h>
#include <torch/script.h>
#include <cub/cub.cuh>
#include "../utils.h"
#include "ops.h"

namespace distemb {
namespace cuda {

template <typename T>
__global__ void BinarySearchKernel(T* in, T* range_partition, int64_t num_part,
                                   int64_t local_num_gpus, T* out,
                                   int64_t num_item) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_item) {
    // levearge cub binary search
    T val = in[idx];
    int64_t rank = cub::UpperBound(range_partition, num_part, val);
    out[idx] = (rank - 1) * local_num_gpus + val % local_num_gpus;
  }
}

template <typename T>
__global__ void BinarySearchKernel2(T* in, T* range, int64_t num_part, T* out,
                                    int64_t num_item) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_item) {
    // levearge cub binary search
    T val = in[idx];
    int64_t rank = cub::UpperBound(range, num_part, val);
    out[idx] = rank;
  }
}

template <typename T>
__global__ void ReminderKernel(T* in, int64_t local_num_gpus, T* out,
                               int64_t num_item) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_item) {
    T val = in[idx];
    out[idx] = val % local_num_gpus;
  }
}

torch::Tensor CUDABinarySearch(torch::Tensor in, torch::Tensor range_partition,
                               int64_t local_num_gpus) {
  int64_t num_item = in.size(0);
  int64_t num_part = range_partition.size(0);
  auto out = torch::empty_like(in);

  int64_t block_size = 1024;
  int64_t grid_size = (num_item + block_size - 1) / block_size;
  INTEGER_TYPE_SWITCH(in.scalar_type(), scalar_t, {
    BinarySearchKernel<scalar_t><<<grid_size, block_size>>>(
        in.data_ptr<scalar_t>(), range_partition.data_ptr<scalar_t>(), num_part,
        local_num_gpus, out.data_ptr<scalar_t>(), num_item);
  });
  return out;
}

torch::Tensor CUDABinarySearch2(torch::Tensor in, torch::Tensor range) {
  int64_t num_item = in.size(0);
  int64_t num_part = range.size(0);
  auto out = torch::empty_like(in);

  int64_t block_size = 1024;
  int64_t grid_size = (num_item + block_size - 1) / block_size;
  INTEGER_TYPE_SWITCH(in.scalar_type(), scalar_t, {
    BinarySearchKernel2<scalar_t><<<grid_size, block_size>>>(
        in.data_ptr<scalar_t>(), range.data_ptr<scalar_t>(), num_part,
        out.data_ptr<scalar_t>(), num_item);
  });
  return out;
}

torch::Tensor Reminder(torch::Tensor in, int64_t local_num_gpus) {
  int64_t num_item = in.size(0);
  auto out = torch::empty_like(in);

  int64_t block_size = 1024;
  int64_t grid_size = (num_item + block_size - 1) / block_size;
  INTEGER_TYPE_SWITCH(in.scalar_type(), scalar_t, {
    ReminderKernel<scalar_t>
        <<<grid_size, block_size>>>(in.data_ptr<scalar_t>(), local_num_gpus,
                                    out.data_ptr<scalar_t>(), num_item);
  });
  return out;
}
}  // namespace cuda
}  // namespace distemb