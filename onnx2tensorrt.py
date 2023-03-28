# CUDA & TensorRT
#import pycuda.driver as cuda 
# from cuda import cuda 
# import pycuda.autoinit

"""
pip install nvidia-pyindex
pip install nvidia-tensorrt
"""
import tensorrt as trt
import sys
import torch

def setup_builder():
  logger = trt.Logger(trt.Logger.ERROR)
  builder = trt.Builder(logger)
  return logger, builder

def onnx2tensorrt(onnx_path, tensorrt_path):
  trt_logger, builder = setup_builder()
  network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  config = builder.create_builder_config()
  # #DeprecationWarning: Use set_memory_pool_limit instead config.max_workspace_size = 256 * 1024 * 1024
  # config.max_workspace_size = 256 * 1024 * 1024
  config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)

  # # allow TensorRT to use up to 1GB of GPU memory for tactic selection
  # builder.max_workspace_size = 1 << 30
  # # we have only one image in batch
  # builder.max_batch_size = 1
  # # use FP16 mode if possible
  # if builder.platform_has_fast_fp16:
  #   builder.fp16_mode = True

  parse = trt.OnnxParser(network, trt_logger)
  onnx_parse_status = parse.parse_from_file(onnx_path)
  if not onnx_parse_status:
    sys.exit("ONNX({}) parse error".format(onnx_path))

  # plan = builder.build_serialized_network(network, config)
  # print(plan)
  # with open(tensorrt_path, "wb") as f:
  #   f.write(plan)

  # Write engine
  engineString = builder.build_serialized_network(network, config)
  if engineString == None:
    print("Failed building engine!")
    return
  with open(tensorrt_path, "wb") as f:
    f.write(engineString)
  print("onnx2tensorrt DONE")


if __name__ == '__main__':
  print("convert to TensorRT")
  ######
  onnx_model_path = "cnn.onnx"
  # tensorrt_model_path = "/content/cnn.plan" 
  tensorrt_model_path = "cnn.plan" 
  print(onnx_model_path, tensorrt_model_path)

  onnx2tensorrt(onnx_model_path, tensorrt_model_path)