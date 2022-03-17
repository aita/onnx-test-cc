#include "onnx.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <onnxruntime_cxx_api.h>

class ONNXRunner {
public:
  ONNXRunner(const std::string &model)
      : session_{env_, model.c_str(), Ort::SessionOptions{nullptr}} {}

  void Run(const int64_t *input_shape, std::size_t input_shape_size,
           const int64_t *output_shape, std::size_t output_shape_size,
           const char *const *input_names, std::size_t input_names_size,
           const char *const *output_names, std::size_t output_names_size,
           float *input, std::size_t input_size, int64_t *output,
           std::size_t output_size) {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input, input_size, input_shape, input_shape_size);
    auto output_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, output, output_size, output_shape, output_shape_size);

    session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor,
                 input_names_size, output_names, &output_tensor,
                 output_names_size);
  }

private:
  Ort::Env env_;
  Ort::Session session_;
};

ONNXRunner *onnx_runner_new(const char *model_path) {
  return new ONNXRunner(model_path);
}

void onnx_runner_free(ONNXRunner *runner) { delete runner; }

void onnx_runner_run(ONNXRunner *runner, const int64_t *input_shape,
                     std::size_t input_shape_size, const int64_t *output_shape,
                     std::size_t output_shape_size,
                     const char *const *input_names,
                     std::size_t input_names_size,
                     const char *const *output_names,
                     std::size_t output_names_size, float *input,
                     std::size_t input_size, int64_t *output,
                     std::size_t output_size) {
  runner->Run(input_shape, input_shape_size, output_shape, output_shape_size,
              input_names, input_names_size, output_names, output_names_size,
              input, input_size, output, output_size);
}