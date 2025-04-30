#include <cuda_runtime.h>
#include <torch/script.h>
#include "../utils.h"
#include "ops.h"

namespace distemb {

namespace cuda {

template <typename DataType, typename IndexType, int WARP_SIZE>
__global__ void UVAFetchKernel(DataType* src, IndexType* src_index,
                               DataType* dst, IndexType* dst_index,
                               int64_t num_elem, int64_t dim) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t warp_id = thread_id / WARP_SIZE;
  int64_t lane = thread_id % WARP_SIZE;
  int64_t warp_num = blockDim.x * gridDim.x / WARP_SIZE;

  for (int i = warp_id; i < num_elem; i += warp_num) {
    int64_t src_idx = src_index[i];
    int64_t dst_idx = dst_index[i];
    for (int j = lane; j < dim; j += WARP_SIZE) {
      dst[dst_idx * dim + j] = src[src_idx * dim + j];
    }
  }
}

void CUDAFetch(torch::Tensor src, torch::Tensor src_index, torch::Tensor dst,
               torch::Tensor dst_index) {
  CHECK(src.size(1) == dst.size(1));
  CHECK(dst_index.numel() == src_index.numel());
  int64_t num_elem = src_index.numel();
  int64_t dim = src.size(1);

  FLOAT_TYPE_SWITCH(src.dtype(), DataType, {
    INTEGER_TYPE_SWITCH(src_index.dtype(), IndexType, {
      int block_size = 256;
      int grid_size = (num_elem + block_size - 1) / block_size;
      UVAFetchKernel<DataType, IndexType, 32><<<grid_size, block_size>>>(
          src.data_ptr<DataType>(), src_index.data_ptr<IndexType>(),
          dst.data_ptr<DataType>(), dst_index.data_ptr<IndexType>(), num_elem,
          dim);
    });
  });
}

template <typename DataType, typename IndexType, int WARP_SIZE>
__global__ void UVAFetchKernel2(DataType* src, IndexType* src_index,
                                DataType* dst, int64_t num_elem, int64_t dim) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t warp_id = thread_id / WARP_SIZE;
  int64_t lane = thread_id % WARP_SIZE;
  int64_t warp_num = blockDim.x * gridDim.x / WARP_SIZE;

  for (int i = warp_id; i < num_elem; i += warp_num) {
    int64_t src_idx = src_index[i];
    for (int j = lane; j < dim; j += WARP_SIZE) {
      dst[i * dim + j] = src[src_idx * dim + j];
    }
  }
}

template <typename DataType, typename IndexType>
__global__ void UVAFetch1DKernel2(DataType* src, IndexType* src_index,
                                  DataType* dst, int64_t num_elem) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id < num_elem) {
    dst[thread_id] = src[src_index[thread_id]];
  }
}

void CUDAFetch2(torch::Tensor src, torch::Tensor src_index, torch::Tensor dst) {
  CHECK(src.sizes().size() == 1 || src.sizes().size() == 2);
  CHECK(src.sizes().size() == dst.sizes().size());
  int64_t dim;
  if (src.sizes().size() == 1) {
    dim = 1;
  } else {
    CHECK(src.size(1) == dst.size(1));
    dim = src.size(1);
  }

  int64_t num_elem = src_index.numel();
  if (dim > 1) {
    FLOAT_TYPE_SWITCH(src.dtype(), DataType, {
      INTEGER_TYPE_SWITCH(src_index.dtype(), IndexType, {
        int block_size = 256;
        int grid_size = (num_elem + block_size - 1) / block_size;
        UVAFetchKernel2<DataType, IndexType, 32><<<grid_size, block_size>>>(
            src.data_ptr<DataType>(), src_index.data_ptr<IndexType>(),
            dst.data_ptr<DataType>(), num_elem, dim);
      });
    });
  } else {
    FLOAT_TYPE_SWITCH(src.dtype(), DataType, {
      INTEGER_TYPE_SWITCH(src_index.dtype(), IndexType, {
        int64_t block_size = 1024;
        int64_t grid_size = (num_elem + block_size - 1) / block_size;
        UVAFetch1DKernel2<DataType, IndexType><<<grid_size, block_size>>>(
            src.data_ptr<DataType>(), src_index.data_ptr<IndexType>(),
            dst.data_ptr<DataType>(), num_elem);
      });
    });
  }
}

template <typename DataType, typename IndexType, int WARP_SIZE>
__global__ void CUDAFetchWithCacheKernel2(DataType* cpu_src, DataType* gpu_src,
                                          IndexType* index,
                                          IndexType* index_map, DataType* dst,
                                          int64_t num_elem, int64_t dim) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t warp_id = thread_id / WARP_SIZE;
  int64_t lane = thread_id % WARP_SIZE;
  int64_t warp_num = blockDim.x * gridDim.x / WARP_SIZE;

  for (int i = warp_id; i < num_elem; i += warp_num) {
    int64_t idx = index[i];
    int64_t gpu_idx = index_map[idx];
    if (gpu_idx != -1) {
      for (int j = lane; j < dim; j += WARP_SIZE) {
        dst[i * dim + j] = gpu_src[gpu_idx * dim + j];
      }
    } else {
      for (int j = lane; j < dim; j += WARP_SIZE) {
        dst[i * dim + j] = cpu_src[idx * dim + j];
      }
    }
  }
}

template <typename DataType, typename IndexType>
__global__ void CUDAFetchWithCache1DKernel2(DataType* cpu_src,
                                            DataType* gpu_src, IndexType* index,
                                            IndexType* index_map, DataType* dst,
                                            int64_t num_elem) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_id < num_elem) {
    int64_t idx = index[thread_id];
    int64_t gpu_idx = index_map[idx];
    if (gpu_idx != -1) {
      dst[thread_id] = gpu_src[gpu_idx];
    } else {
      dst[thread_id] = cpu_src[idx];
    }
  }
}

void CUDAFetchWithCache2(torch::Tensor cpu_src, torch::Tensor gpu_src,
                         torch::Tensor src_index, torch::Tensor index_map,
                         torch::Tensor dst) {
  CHECK(gpu_src.sizes().size() == 1 || gpu_src.sizes().size() == 2);
  CHECK(gpu_src.sizes().size() == cpu_src.sizes().size());
  CHECK(gpu_src.sizes().size() == dst.sizes().size());
  int64_t dim;
  if (gpu_src.sizes().size() == 1) {
    dim = 1;
  } else {
    CHECK(gpu_src.size(1) == cpu_src.size(1));
    CHECK(gpu_src.size(1) == dst.size(1));
    dim = gpu_src.size(1);
  }

  int64_t num_elem = src_index.numel();
  if (dim > 1) {
    FLOAT_TYPE_SWITCH(gpu_src.dtype(), DataType, {
      INTEGER_TYPE_SWITCH(src_index.dtype(), IndexType, {
        int block_size = 256;
        int grid_size = (num_elem + block_size - 1) / block_size;
        CUDAFetchWithCacheKernel2<DataType, IndexType, 32>
            <<<grid_size, block_size>>>(
                cpu_src.data_ptr<DataType>(), gpu_src.data_ptr<DataType>(),
                src_index.data_ptr<IndexType>(),
                index_map.data_ptr<IndexType>(), dst.data_ptr<DataType>(),
                num_elem, dim);
      });
    });
  } else {
    FLOAT_TYPE_SWITCH(gpu_src.dtype(), DataType, {
      INTEGER_TYPE_SWITCH(src_index.dtype(), IndexType, {
        int64_t block_size = 1024;
        int64_t grid_size = (num_elem + block_size - 1) / block_size;
        CUDAFetchWithCache1DKernel2<DataType, IndexType>
            <<<grid_size, block_size>>>(cpu_src.data_ptr<DataType>(),
                                        gpu_src.data_ptr<DataType>(),
                                        src_index.data_ptr<IndexType>(),
                                        index_map.data_ptr<IndexType>(),
                                        dst.data_ptr<DataType>(), num_elem);
      });
    });
  }
}

template <typename DataType, typename IndexType, int WARP_SIZE>
__global__ void UVACombineKernel(DataType* src, DataType* dst,
                                 IndexType* dst_index, int64_t num_elem,
                                 int64_t dim) {
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t warp_id = thread_id / WARP_SIZE;
  int64_t lane = thread_id % WARP_SIZE;
  int64_t warp_num = blockDim.x * gridDim.x / WARP_SIZE;

  for (int i = warp_id; i < num_elem; i += warp_num) {
    int64_t src_idx = i;
    int64_t dst_idx = dst_index[i];
    for (int j = lane; j < dim; j += WARP_SIZE) {
      dst[dst_idx * dim + j] = src[src_idx * dim + j];
    }
  }
}

void CUDACombine(torch::Tensor src, torch::Tensor dst,
                 torch::Tensor dst_index) {
  CHECK(src.size(1) == dst.size(1));
  int64_t num_elem = dst_index.numel();
  int64_t dim = src.size(1);

  FLOAT_TYPE_SWITCH(src.dtype(), DataType, {
    INTEGER_TYPE_SWITCH(dst_index.dtype(), IndexType, {
      int block_size = 256;
      int grid_size = (num_elem + block_size - 1) / block_size;
      UVACombineKernel<DataType, IndexType, 32><<<grid_size, block_size>>>(
          src.data_ptr<DataType>(), dst.data_ptr<DataType>(),
          dst_index.data_ptr<IndexType>(), num_elem, dim);
    });
  });
}

}  // namespace cuda

}  // namespace distemb
