#include <cuda_runtime.h>
#include <torch/script.h>
#include "../utils.h"
#include "ops.h"

namespace distemb {

namespace cuda {

template <typename DataType, typename IndexType, int WARP_SIZE>
__global__ void UVASetKernel(DataType* src, DataType* dst, IndexType* dst_index,
                             int64_t num_elem, int64_t dim) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t warp_id = thread_id / WARP_SIZE;
  int64_t lane = thread_id % WARP_SIZE;
  int64_t warp_num = blockDim.x * gridDim.x / WARP_SIZE;

  for (int i = warp_id; i < num_elem; i += warp_num) {
    int64_t dst_idx = dst_index[i];
    for (int j = lane; j < dim; j += WARP_SIZE) {
      dst[dst_idx * dim + j] = src[i * dim + j];
    }
  }
}

template <typename DataType, typename IndexType>
__global__ void UVASet1DKernel(DataType* src, DataType* dst,
                               IndexType* dst_index, int64_t num_elem) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id < num_elem) {
    dst[dst_index[thread_id]] = src[thread_id];
  }
}

void CUDASet(torch::Tensor src, torch::Tensor dst, torch::Tensor dst_index) {
  CHECK(src.sizes().size() == 1 || src.sizes().size() == 2);
  CHECK(src.sizes().size() == dst.sizes().size());
  int64_t dim;
  if (src.sizes().size() == 1) {
    dim = 1;
  } else {
    CHECK(src.size(1) == dst.size(1));
    dim = src.size(1);
  }

  int64_t num_elem = dst_index.numel();
  if (dim > 1) {
    FLOAT_TYPE_SWITCH(src.dtype(), DataType, {
      INTEGER_TYPE_SWITCH(dst_index.dtype(), IndexType, {
        int block_size = 256;
        int grid_size = (num_elem + block_size - 1) / block_size;
        UVASetKernel<DataType, IndexType, 32><<<grid_size, block_size>>>(
            src.data_ptr<DataType>(), dst.data_ptr<DataType>(),
            dst_index.data_ptr<IndexType>(), num_elem, dim);
      });
    });
  } else {
    FLOAT_TYPE_SWITCH(src.dtype(), DataType, {
      INTEGER_TYPE_SWITCH(dst_index.dtype(), IndexType, {
        int64_t block_size = 1024;
        int64_t grid_size = (num_elem + block_size - 1) / block_size;
        UVASet1DKernel<DataType, IndexType><<<grid_size, block_size>>>(
            src.data_ptr<DataType>(), dst.data_ptr<DataType>(),
            dst_index.data_ptr<IndexType>(), num_elem);
      });
    });
  }
}

template <typename DataType, typename IndexType, int WARP_SIZE>
__global__ void CUDASetWithCacheKernel(DataType* src, DataType* cpu_dst,
                                       DataType* gpu_dst, IndexType* index,
                                       IndexType* index_map, int64_t num_elem,
                                       int64_t dim) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t warp_id = thread_id / WARP_SIZE;
  int64_t lane = thread_id % WARP_SIZE;
  int64_t warp_num = blockDim.x * gridDim.x / WARP_SIZE;

  for (int i = warp_id; i < num_elem; i += warp_num) {
    int64_t idx = index[i];
    int64_t gpu_idx = index_map[idx];
    if (gpu_idx != -1) {
      for (int j = lane; j < dim; j += WARP_SIZE) {
        gpu_dst[gpu_idx * dim + j] = src[i * dim + j];
      }
    } else {
      for (int j = lane; j < dim; j += WARP_SIZE) {
        cpu_dst[idx * dim + j] = src[i * dim + j];
      }
    }
  }
}

template <typename DataType, typename IndexType>
__global__ void CUDASetWithCache1DKernel(DataType* src, DataType* cpu_dst,
                                         DataType* gpu_dst, IndexType* index,
                                         IndexType* index_map,
                                         int64_t num_elem) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id < num_elem) {
    int64_t idx = index[thread_id];
    int64_t gpu_idx = index_map[idx];
    if (gpu_idx != -1) {
      gpu_dst[gpu_idx] = src[thread_id];
    } else {
      cpu_dst[idx] = src[thread_id];
    }
  }
}

void CUDASetWithCache(torch::Tensor src, torch::Tensor cpu_dst,
                      torch::Tensor gpu_dst, torch::Tensor dst_index,
                      torch::Tensor index_map) {
  CHECK(gpu_dst.sizes().size() == 1 || gpu_dst.sizes().size() == 2);
  CHECK(gpu_dst.sizes().size() == cpu_dst.sizes().size());
  CHECK(gpu_dst.sizes().size() == src.sizes().size());
  int64_t dim;
  if (gpu_dst.sizes().size() == 1) {
    dim = 1;
  } else {
    CHECK(gpu_dst.size(1) == cpu_dst.size(1));
    CHECK(gpu_dst.size(1) == src.size(1));
    dim = gpu_dst.size(1);
  }

  int64_t num_elem = dst_index.numel();
  if (dim > 1) {
    FLOAT_TYPE_SWITCH(gpu_dst.dtype(), DataType, {
      INTEGER_TYPE_SWITCH(dst_index.dtype(), IndexType, {
        int block_size = 256;
        int grid_size = (num_elem + block_size - 1) / block_size;
        CUDASetWithCacheKernel<DataType, IndexType, 32>
            <<<grid_size, block_size>>>(
                src.data_ptr<DataType>(), cpu_dst.data_ptr<DataType>(),
                gpu_dst.data_ptr<DataType>(), dst_index.data_ptr<IndexType>(),
                index_map.data_ptr<IndexType>(), num_elem, dim);
      });
    });
  } else {
    FLOAT_TYPE_SWITCH(gpu_dst.dtype(), DataType, {
      INTEGER_TYPE_SWITCH(dst_index.dtype(), IndexType, {
        int64_t block_size = 1024;
        int64_t grid_size = (num_elem + block_size - 1) / block_size;
        CUDASetWithCache1DKernel<DataType, IndexType>
            <<<grid_size, block_size>>>(
                src.data_ptr<DataType>(), cpu_dst.data_ptr<DataType>(),
                gpu_dst.data_ptr<DataType>(), dst_index.data_ptr<IndexType>(),
                index_map.data_ptr<IndexType>(), num_elem);
      });
    });
  }
}

}  // namespace cuda

}  // namespace distemb
