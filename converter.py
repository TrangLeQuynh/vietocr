import matplotlib.pyplot as plt
from PIL import Image
import torch
import onnxruntime
import numpy as np
import onnx
from vietocr.tool.translate import *
from vietocr.tool.config import Cfg
import argparse
import torch


def read_model(config):
  # config['cnn']['pretrained']=False
  model, vocab = build_model(config)
  device = config['device']
  weights = config['weights']

  model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
  model = model.eval()
  return model


# But before verifying the model’s output with ONNX Runtime, we will check the ONNX model with ONNX’s API. 
def verify_onnx_model(model_path):
  # Load the ONNX model
  onnx_model = onnx.load(model_path)

  # Check that the model is well formed
  onnx.checker.check_model(onnx_model) # verify the model’s structure and confirm that the model has a valid schema

  # Print a human readable representation of the graph
  print(onnx.helper.printable_graph(onnx_model.graph))

def convert_cnn_part(img, save_path, model): 
  # dynamic_axes={
  #   'img': {
  #     3: 'lenght'
  #   }, 
  #   'output': {
  #     0: 'channel',
  #   }
  # }
  with torch.no_grad(): 
    src = model.cnn(img)
    torch.onnx.export(
      model.cnn, 
      img, 
      save_path, 
      export_params=True, 
      opset_version=12, 
      do_constant_folding=True, #wether to execute constant folding for optimization
      verbose=True, 
      input_names=['img'], 
      output_names=['output'], 
      dynamic_axes={
        'img': {3: 'length'}, 
        'output': {0: 'channel'}
      }
    )
  
  return src
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', required=True, help='foo help')  
  args = parser.parse_args()
  config = Cfg.load_config_from_file(args.config)

  save_path = 'cnn.onnx'
  # print(config)
  # print("=============================================")
  model = read_model(config)

  img = torch.rand(1, 3, 32, 170)
  src = convert_cnn_part(img, save_path, model)
  # print(src.shape)

  verify_onnx_model(save_path)
  cnn_session = onnxruntime.InferenceSession(save_path)
  print("=============================================")

