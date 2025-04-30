#pragma once

#include <cuda_runtime.h>
#include <torch/script.h>
#include <sstream>
#include <string>
#include <vector>

namespace distemb {

#define AlignUp(X, ALIGN_SIZE) (((X) + (ALIGN_SIZE)-1) / (ALIGN_SIZE))
#define CUDA_CALL(call)                                                  \
  {                                                                      \
    cudaError_t cudaStatus = call;                                       \
    if (cudaSuccess != cudaStatus) {                                     \
      fprintf(stderr,                                                    \
              "%s:%d ERROR: CUDA RT call \"%s\" failed "                 \
              "with "                                                    \
              "%s (%d).\n",                                              \
              __FILE__, __LINE__, #call, cudaGetErrorString(cudaStatus), \
              cudaStatus);                                               \
      exit(cudaStatus);                                                  \
    }                                                                    \
  }

#define FLOAT_TYPE_SWITCH(val, Type, ...)               \
  do {                                                  \
    if ((val) == torch::kFloat) {                       \
      typedef float Type;                               \
      { __VA_ARGS__ }                                   \
    } else if ((val) == torch::kDouble) {               \
      typedef double Type;                              \
      { __VA_ARGS__ }                                   \
    } else {                                            \
      LOG(FATAL) << "Type can only be float or double"; \
    }                                                   \
  } while (0);

#define INTEGER_TYPE_SWITCH(val, IdType, ...)        \
  do {                                               \
    if ((val) == torch::kInt32) {                    \
      typedef int32_t IdType;                        \
      { __VA_ARGS__ }                                \
    } else if ((val) == torch::kInt64) {             \
      typedef int64_t IdType;                        \
      { __VA_ARGS__ }                                \
    } else {                                         \
      LOG(FATAL) << "ID can only be int32 or int64"; \
    }                                                \
  } while (0);

inline std::vector<std::string> split_string(std::string str, char delimiter) {
  std::vector<std::string> output;
  std::istringstream ss(str);
  std::string sub;
  while (std::getline(ss, sub, delimiter)) {
    output.push_back(sub);
  }
  return output;
}

inline std::string vector2string(std::vector<int64_t> input) {
  std::string output = "";
  for (auto item : input) {
    output += std::to_string(item) + " ";
  }
  return output;
}

inline std::vector<int64_t> string2vector(std::string input) {
  std::vector<int64_t> output;
  auto split_input = split_string(input, ' ');
  for (auto item : split_input) {
    output.push_back(std::stol(item));
  }
  return output;
}

inline std::string dtype2string(torch::ScalarType type) {
  if (type == torch::kInt32) {
    return "int32";
  } else if (type == torch::kInt64) {
    return "int64";
  } else if (type == torch::kFloat) {
    return "float32";
  } else if (type == torch::kDouble) {
    return "float64";
  } else if (type == torch::kBool) {
    return "bool";
  } else {
    fprintf(stderr, "Error in dtype2string, unsupported type!\n");
    exit(-1);
  }
}

inline torch::ScalarType string2dtype(std::string type) {
  if (type == "int32") {
    return torch::kInt32;
  } else if (type == "int64") {
    return torch::kInt64;
  } else if (type == "float32") {
    return torch::kFloat;
  } else if (type == "float64") {
    return torch::kDouble;
  } else if (type == "bool") {
    return torch::kBool;
  } else {
    fprintf(stderr, "Error in string2dtype, unsupported type string!\n");
    exit(-1);
  }
}

}  // namespace distemb