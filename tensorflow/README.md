# TensorFlow

## Environment Preparation

Follow this guide to properly install TensorFlow with pip: <https://www.tensorflow.org/install/pip#step-by-step_instructions>.
Please note that CPU-only version installation is as easy as `pip install "tensorflow>=2"` and you can use virtualenv or conda.
However, the GPU-enabled version installation must be in conda.

## Error Fix for GPU-enabled TF

If you are facing this error when using TensorFlow with GPU:

```text
2023-05-06 16:19:54.078251: W tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.cc:530] Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice. This may result in compilation or runtime failures, if the program we try to run uses routines from libdevice.
Searched for CUDA in the following directories:
  ./cuda_sdk_lib
  /usr/local/cuda-11.8
  /usr/local/cuda
  .
You can choose the search directory by setting xla_gpu_cuda_data_dir in HloModule's DebugOptions.  For most apps, setting the environment variable XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda will work.
2023-05-06 16:19:54.078575: W tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.cc:274] libdevice is required by this HLO module but was not found at ./libdevice.10.bc
2023-05-06 16:19:54.079002: W tensorflow/core/framework/op_kernel.cc:1830] OP_REQUIRES failed at xla_ops.cc:362 : INTERNAL: libdevice not found at ./libdevice.10.bc
```

Consider to downgrade to `tensorflow>=2,<=2.10.1` by doing `pip install -U "tensorflow>=2,<=2.10.1"`. See this issue for details: <https://github.com/tensorflow/tensorflow/issues/56927#issuecomment-1327625131>.
