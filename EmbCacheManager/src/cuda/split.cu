#include <cuda_runtime.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <torch/script.h>
#include <vector>
#include "ops.h"

namespace distemb {

namespace cuda {

// for mask, 0: local, 1: remote_hot, 2: remote_cold
template <typename IdType>
std::vector<torch::Tensor> _split(torch::Tensor ids, torch::Tensor mask) {
  int64_t num_elem = ids.numel();

  torch::Tensor local_mask = torch::zeros(
      {num_elem},
      torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
  torch::Tensor remote_hot_mask = torch::zeros(
      {num_elem},
      torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
  torch::Tensor remote_cold_mask = torch::zeros(
      {num_elem},
      torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

  auto ids_ptr = ids.data_ptr<IdType>();
  auto mask_ptr = mask.data_ptr<int8_t>();
  auto local_mask_ptr = local_mask.data_ptr<bool>();
  auto remote_hot_mask_ptr = remote_hot_mask.data_ptr<bool>();
  auto remote_cold_mask_ptr = remote_cold_mask.data_ptr<bool>();

  thrust::for_each(thrust::device, thrust::make_counting_iterator<int64_t>(0),
                   thrust::make_counting_iterator<int64_t>(num_elem),
                   [=] __device__(int64_t i) {
                     int64_t id = ids_ptr[i];
                     int8_t value = mask_ptr[id];
                     if (value == 0) {
                       local_mask_ptr[i] = true;
                     } else if (value == 1) {
                       remote_hot_mask_ptr[i] = true;
                     } else if (value == 2) {
                       remote_cold_mask_ptr[i] = true;
                     }
                   });

  // torch::Tensor local_ids = ids.masked_select(local_mask);
  // torch::Tensor remote_hot_ids = ids.masked_select(remote_hot_mask);
  // torch::Tensor remote_cold_ids = ids.masked_select(remote_cold_mask);

  return {local_mask, remote_hot_mask, remote_cold_mask};
}

std::vector<torch::Tensor> splitCUDA(torch::Tensor ids, torch::Tensor mask) {
  return AT_DISPATCH_ALL_TYPES(ids.scalar_type(), "splitCUDA",
                               [&] { return _split<scalar_t>(ids, mask); });
}

}  // namespace cuda
}  // namespace distemb