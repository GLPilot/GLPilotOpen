#pragma once
#include <omp.h>
// #include <pthread.h>
#include <atomic>
#include <mutex>
#include <vector>

#pragma GCC push_options
#pragma GCC optimize("Ofast,unroll-loops")
namespace distemb {

template <typename Key, int SIZE>
struct CacheLine {
  // this cache line only use to judge wheather the key is in the cache line
  // If the key is in the cache line, the location for this key is (the idx of
  // the key + CacheLine idx * SIZE);

  // mutex are used by the caller to protect the cache line
  // pthread_mutex_t mutex;
  // pthread_mutexattr_t mutexattr;
  omp_lock_t lock;

  // -1 for empty in data;
  Key data[SIZE];

  // time count for lru
  std::atomic<int> access_time_table[SIZE];

  CacheLine() {
    for (int i = 0; i < SIZE; i++) {
      data[i] = -1;
      access_time_table[i] = 0;
    }

    // pthread_mutexattr_init(&mutexattr);
    // pthread_mutexattr_settype(&mutexattr, PTHREAD_PROCESS_SHARED);
    // pthread_mutex_init(&mutex, &mutexattr);

    // init omp lock
    omp_init_lock(&lock);
  }

  ~CacheLine() {
    // pthread_mutex_destroy(&mutex);
    // pthread_mutexattr_destroy(&mutexattr);
    omp_destroy_lock(&lock);
  }

  // read operation, return the idx of the key in the cache line
  // if the key is not in the cache line, return -1
  inline int query(const Key key) {
    for (int i = 0; i < SIZE; i++) {
      if (data[i] == key) {
        increase_time_table();
        access_time_table[i] = 0;
        return i;
      }
    }
    return -1;
  }

  // update operation, return the idx of the key in the cache line
  // if the key is not inserted, return -1
  inline int update(const Key key) {
    int idx;
    bool exist;
    idx = find_or_empty(key, exist);
    if (idx >= 0) {
      if (!exist) data[idx] = key;
      access_time_table[idx] = 0;
      return idx;
    }

    // use lru to replace the key
    idx = find_lru();
    if (idx >= 0) {
      data[idx] = key;
      access_time_table[idx] = 0;
      return idx;
    }
    // update failed
    return -1;
  }

  inline void reset(const int idx) { data[idx] = -1; }

  inline int find_or_empty(const Key key, bool& exist) {
    int idx = -1;
    exist = false;
    for (int i = 0; i < SIZE; i++) {
      auto value = data[i];
      if (value == key) {
        idx = i;
        exist = true;
        return idx;  // find
      } else if (value == -1) {
        idx = i;
      }
    }
    return idx;
  }

  inline int find_lru() {
    int lru_idx = 0;
    int lru_time = access_time_table[0];
    for (int i = 1; i < SIZE; i++) {
      if (access_time_table[i] > lru_time) {
        lru_time = access_time_table[i];
        lru_idx = i;
      }
    }
    // if the lru time is the same as the 0, it means that all the
    // keys in the cache line are new data, so we can't find the lru key
    return lru_time == 0 ? -1 : lru_idx;
  }

  void debug_info() {
    printf("-----------------------------\n");
    printf("data: ");
    for (int i = 0; i < SIZE; i++) {
      printf("%ld ", data[i]);
    }
    printf("access_time_table: ");
    for (int i = 0; i < SIZE; i++) {
      // get value from atomic
      int value = access_time_table[i];
      printf("%d ", value);
    }
    printf("\n");
    printf("-----------------------------\n");
  }

  void increase_time_table() {
    for (int i = 0; i < SIZE; i++) {
      access_time_table[i]++;
    }
  }
};

}  // namespace distemb

#pragma GCC pop_options