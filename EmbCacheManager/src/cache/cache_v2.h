#pragma once
#include <torch/script.h>

#include "../shm/shared_memory.h"
#include "cache_line_v2.h"

/*
#define CHECK_PTHREAD_RET(ret, msg) \
  if (ret != 0) {                   \
    perror(msg);                    \
    exit(EXIT_FAILURE);             \
  }
*/

namespace distemb {
template <typename Key, typename IdxType>
class SACacheIndex_v2 {
  int static constexpr LINE_SIZE = 16;
  using CacheLineType = CacheLine<Key, LINE_SIZE>;

 public:
  SACacheIndex_v2(int64_t capacity, std::string fname, bool create) {
    create_ = create;
    shared_mem_ = std::make_shared<SharedMemory>(fname, false);
    num_cache_lines_ = (capacity + LINE_SIZE - 1) / LINE_SIZE;

    size_t shared_mem_size = sizeof(CacheLineType) * num_cache_lines_;
    char* mem;
    if (create)
      mem = static_cast<char*>(shared_mem_->CreateNew(shared_mem_size));
    else
      mem = static_cast<char*>(shared_mem_->Open(shared_mem_size));

    if (create) {
      cache_lines_ = reinterpret_cast<CacheLineType*>(mem);
      for (int64_t i = 0; i < num_cache_lines_; i++)
        new (cache_lines_ + i) CacheLineType();
    } else {
      cache_lines_ = reinterpret_cast<CacheLineType*>(mem);
    }
  }

  // add key to cache
  // return position of the key in the cache
  torch::Tensor add(torch::Tensor keys) {
    torch::Tensor locs =
        torch::full_like(keys, -1,
                         torch::TensorOptions()
                             .dtype(torch::CppTypeToScalarType<IdxType>())
                             .device(torch::kCPU));

    int64_t numel = keys.numel();
    auto key_ptr = keys.data_ptr<Key>();
    auto locs_ptr = locs.data_ptr<IdxType>();

#pragma omp parallel for
    for (int64_t i = 0; i < numel; i++) {
      Key key = key_ptr[i];
      int64_t cacheline_id = key % num_cache_lines_;
      CacheLineType& cacheline = cache_lines_[cacheline_id];

      // CHECK_PTHREAD_RET(pthread_mutex_lock(&cacheline.mutex),
      //                   "lock fail in add");
      omp_set_lock(&cacheline.lock);
      int loc = cacheline.update(key);
      // CHECK_PTHREAD_RET(pthread_mutex_unlock(&cacheline.mutex),
      //                   "unlock fail in add");
      omp_unset_lock(&cacheline.lock);

      if (loc != -1) locs_ptr[i] = cacheline_id * LINE_SIZE + loc;
    }

    return locs;
  }

  torch::Tensor query(torch::Tensor keys) {
    torch::Tensor locs =
        torch::full_like(keys, -1,
                         torch::TensorOptions()
                             .dtype(torch::CppTypeToScalarType<IdxType>())
                             .device(torch::kCPU));

    int64_t numel = keys.numel();
    auto key_ptr = keys.data_ptr<Key>();
    auto locs_ptr = locs.data_ptr<IdxType>();

#pragma omp parallel for
    for (int64_t i = 0; i < numel; i++) {
      Key key = key_ptr[i];
      int64_t cacheline_id = key % num_cache_lines_;
      CacheLineType& cacheline = cache_lines_[cacheline_id];
      int loc = cacheline.query(key);
      if (loc != -1) locs_ptr[i] = cacheline_id * LINE_SIZE + loc;
    }

    return locs;
  }

  torch::Tensor get_all_entries() {
    torch::Tensor entries =
        torch::empty({num_cache_lines_ * LINE_SIZE},
                     torch::TensorOptions()
                         .dtype(torch::CppTypeToScalarType<Key>())
                         .device(torch::kCPU));
    auto entries_ptr = entries.data_ptr<Key>();

#pragma omp parallel for
    for (int64_t i = 0; i < num_cache_lines_; i++) {
      for (int j = 0; j < LINE_SIZE; j++) {
        entries_ptr[i * LINE_SIZE + j] = cache_lines_[i].data[j];
      }
    }
    return entries;
  }

  torch::Tensor get_entries_by_index(torch::Tensor index) {
    auto index_ptr = index.data_ptr<IdxType>();
    int64_t numel = index.numel();
    torch::Tensor entries =
        torch::empty({numel}, torch::TensorOptions()
                                  .dtype(torch::CppTypeToScalarType<Key>())
                                  .device(torch::kCPU));
    auto entries_ptr = entries.data_ptr<Key>();

#pragma omp parallel for
    for (int64_t i = 0; i < numel; i++) {
      IdxType loc = index_ptr[i];
      int64_t cacheline_id = loc / LINE_SIZE;
      int loc_in_cacheline = loc % LINE_SIZE;
      entries_ptr[i] = cache_lines_[cacheline_id].data[loc_in_cacheline];
    }

    return entries;
  }

  // reset target position to invalid
  void reset(torch::Tensor index) {
    auto index_ptr = index.data_ptr<IdxType>();
    int64_t numel = index.numel();

#pragma omp parallel for
    for (int64_t i = 0; i < numel; i++) {
      IdxType loc = index_ptr[i];
      int64_t cacheline_id = loc / LINE_SIZE;
      int loc_in_cacheline = loc % LINE_SIZE;
      cache_lines_[cacheline_id].reset(loc_in_cacheline);
    }
  }

  void debug_info(int64_t index) { cache_lines_[index].debug_info(); }

  int64_t get_capacity() const { return num_cache_lines_ * LINE_SIZE; }

  // void lock(int64_t cacheline_id, int64_t second) {
  //   CacheLineType& cacheline = cache_lines_[cacheline_id];
  //
  //  pthread_mutex_lock(&cacheline.mutex);
  //  std::this_thread::sleep_for(std::chrono::seconds(second));
  //  pthread_mutex_unlock(&cacheline.mutex);
  //}

  ~SACacheIndex_v2() {
    if (cache_lines_) shared_mem_->Release();
  }

 private:
  CacheLineType* cache_lines_ = nullptr;
  std::shared_ptr<SharedMemory> shared_mem_ = nullptr;
  int64_t num_cache_lines_;
  bool create_;
};
}  // namespace distemb