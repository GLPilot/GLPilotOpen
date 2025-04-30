#pragma once

#include <torch/script.h>

namespace distemb {

namespace cuda {

std::vector<torch::Tensor> splitCUDA(torch::Tensor ids, torch::Tensor mask);

void PinTensor(torch::Tensor tensor);

void UnpinTensor(torch::Tensor tensor);

void CUDAFetch(torch::Tensor src, torch::Tensor src_index, torch::Tensor dst,
               torch::Tensor dst_index);

void CUDAFetch2(torch::Tensor src, torch::Tensor src_index, torch::Tensor dst);

void CUDAFetchWithCache2(torch::Tensor cpu_src, torch::Tensor gpu_src,
                         torch::Tensor src_index, torch::Tensor index_map,
                         torch::Tensor dst);

void CUDASet(torch::Tensor src, torch::Tensor dst, torch::Tensor dst_index);

void CUDASetWithCache(torch::Tensor src, torch::Tensor cpu_dst,
                      torch::Tensor gpu_dst, torch::Tensor dst_index,
                      torch::Tensor index_map);

void CUDACombine(torch::Tensor src, torch::Tensor dst, torch::Tensor dst_index);

void RegisterPtr(int64_t ptr, int64_t size);

void UnregisterPtr(int64_t ptr);

torch::Tensor CUDABinarySearch(torch::Tensor in, torch::Tensor range_partition,
                               int64_t local_num_gpus);

torch::Tensor CUDABinarySearch2(torch::Tensor in, torch::Tensor range);

torch::Tensor Reminder(torch::Tensor in, int64_t local_num_gpus);

}  // namespace cuda
}  // namespace distemb