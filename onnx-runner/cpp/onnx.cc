#include "onnx.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <onnxruntime_cxx_api.h>

void iris_test() {
  Ort::Env env;
  Ort::Session session{env, "iris.onnx", Ort::SessionOptions{nullptr}};

  std::array<float, 10 * 4> input = {
      6.1, 2.8, 4.7, 1.2, 5.7, 3.8, 1.7, 0.3, 7.7, 2.6, 6.9, 2.3, 6.,  2.9,
      4.5, 1.5, 6.8, 2.8, 4.8, 1.4, 5.4, 3.4, 1.5, 0.4, 5.6, 2.9, 3.6, 1.3,
      6.9, 3.1, 5.1, 2.3, 6.2, 2.2, 4.5, 1.5, 5.8, 2.7, 3.9, 1.2,
  };
  std::array<int64_t, 10> result{};

  std::array<int64_t, 2> input_shape{10, 4};
  std::array<int64_t, 1> output_shape{10};

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  auto input_tensor =
      Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(),
                                      input_shape.data(), input_shape.size());
  auto output_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, result.data(), result.size(), output_shape.data(),
      output_shape.size());

  std::array<const char *, 1> input_names = {"inputs"};
  std::array<const char *, 1> output_names = {"label"};

  session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor,
              input_names.size(), output_names.data(), &output_tensor,
              output_names.size());

  std::cout << "Result: ";
  for (auto i : result) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
}
