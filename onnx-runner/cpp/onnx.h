#ifndef ONNX_RUNNER_H
#define ONNX_RUNNER_H
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ONNXRunner;
typedef struct ONNXRunner ONNXRunner;

ONNXRunner *onnx_runner_new(const char *model_path);
void onnx_runner_free(ONNXRunner *runner);
void onnx_runner_run(ONNXRunner *runner, const int64_t *input_shape,
                     size_t input_shape_size, const int64_t *output_shape,
                     size_t output_shape_size, const char *const *input_names,
                     size_t input_names_size, const char *const *output_names,
                     size_t output_names_size, float *input, size_t input_size,
                     int64_t *output, size_t output_size);

#ifdef __cplusplus
}
#endif

#endif