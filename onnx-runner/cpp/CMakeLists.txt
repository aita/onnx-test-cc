cmake_minimum_required(VERSION 3.10)

# set the project name
project(onnx-runner LANGUAGES CXX C)

# add the executable
add_library(onnx-runner STATIC onnx.cc onnx.h)

include(FetchContent)
FetchContent_Declare(
  onnxruntime
  URL  https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-linux-x64-1.10.0.tgz
)
FetchContent_MakeAvailable(onnxruntime)

set(ONNXRUNTIME_DIR ${onnxruntime_SOURCE_DIR})
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
set(ONNXRUNTIME_INCLUDE ${ONNXRUNTIME_DIR}/include)

target_include_directories(onnx-runner PRIVATE ${ONNXRUNTIME_INCLUDE})
target_link_libraries(onnx-runner PRIVATE ${ONNXRUNTIME_LIB})

install (TARGETS onnx-runner LIBRARY DESTINATION .)
install (DIRECTORY ${ONNXRUNTIME_DIR}/lib DESTINATION .)
install (FILES onnx.h DESTINATION include)
