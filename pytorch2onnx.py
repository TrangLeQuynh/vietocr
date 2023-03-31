# import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from vietocr.tool.translate import *
from vietocr.tool.config import Cfg
import argparse
import torch
# import cv2
from torch.nn.functional import softmax
# ONNX: pip install onnx, onnxruntime
try:
  import onnx
  import onnxruntime
except ImportError as e:
  raise ImportError(f'Please install onnx and onnxruntime first. {e}')

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
  # print(onnx.helper.printable_graph(onnx_model.graph))

def convert_cnn_part(img, save_path, model): 
  with torch.no_grad(): 
    src = model.cnn(img)
    torch.onnx.export(
      model.cnn, 
      img, 
      save_path, 
      export_params=True, 
      opset_version=11, 
      do_constant_folding=True, #wether to execute constant folding for optimization
      verbose=False, 
      input_names=['img'], 
      output_names=['output'], 
      dynamic_axes={
        'img': {3: 'length'}, 
        'output': {0: 'channel'}
      }
    )
  
  return src

"""
  src: timestep x batch_size x channel
  hidden: batch_size x hid_dim
  encoder_outputs: src_len x batch_size x hid_dim
"""
def convert_encoder_part(model, src, save_path): 
  encoder_outputs, hidden = model.transformer.encoder(src) 
  torch.onnx.export(
    model.transformer.encoder, 
    src, 
    save_path, 
    export_params=True, 
    opset_version=11, 
    do_constant_folding=True, 
    input_names=['src'], 
    output_names=['encoder_outputs', 'hidden'], 
    dynamic_axes={
      'src':{0: "channel_input"}, 
      'encoder_outputs': {0: 'channel_output'}
    }
  )
  return (hidden, encoder_outputs)    

"""
  tgt: timestep x batch_size 
  hidden: batch_size x hid_dim
  encouder: src_len x batch_size x hid_dim
  output: batch_size x 1 x vocab_size
"""
def convert_decoder_part(model, tgt, hidden, encoder_outputs, save_path):
  tgt = tgt[-1]
  torch.onnx.export(model.transformer.decoder,
    (tgt, hidden, encoder_outputs),
    save_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['tgt', 'hidden', 'encoder_outputs'],
    output_names=['output', 'hidden_out', 'last'],
    dynamic_axes={
      'encoder_outputs':{0: "channel_input"},
      'last': {0: 'channel_output'}
    })
  
def run_onnx_converter():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', required=True, help='foo help')  
  args = parser.parse_args()
  config = Cfg.load_config_from_file(args.config)

  model = read_model(config)

  img = torch.rand(1, 3, 32, 256)

  save_path = 'cnn.onnx'
  src = convert_cnn_part(img, save_path, model)

  save_path = 'encoder.onnx'
  hidden, encoder_outputs = convert_encoder_part(model, src, save_path)
  tgt = torch.LongTensor([[1] * len(img)])
  save_path = 'decoder.onnx'
  convert_decoder_part(model, tgt, hidden, encoder_outputs, save_path)

def translate_onnx(img, session, max_seq_length=128, sos_token=1, eos_token=2):
  """data: BxCxHxW"""
  cnn_session, encoder_session, decoder_session = session
  
  # create cnn input
  cnn_input = {cnn_session.get_inputs()[0].name: img}
  src = cnn_session.run(None, cnn_input)
  
  # create encoder input
  encoder_input = {encoder_session.get_inputs()[0].name: src[0]}
  encoder_outputs, hidden = encoder_session.run(None, encoder_input)
  translated_sentence = [[sos_token] * len(img)]
  max_length = 0

  while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
    tgt_inp = translated_sentence
    tgt_inp = tgt_inp[-1]

    decoder_input = {
      decoder_session.get_inputs()[0].name: tgt_inp, 
      decoder_session.get_inputs()[1].name: hidden, 
      decoder_session.get_inputs()[2].name: encoder_outputs
    }
    output, hidden, _ = decoder_session.run(None, decoder_input)
    # output = np.expand_dims(output, axis=1)
    output = torch.Tensor(output)
    output = output.unsqueeze(1)  

    output = softmax(output, dim=-1)

    values, indices = torch.topk(output, 1)
    indices = indices[:, -1, 0]
    indices = indices.tolist()

    translated_sentence.append(indices)
    max_length += 1

    del output

  translated_sentence = np.asarray(translated_sentence).T

  return translated_sentence

def test_converter_onnx(show_img = False):
  config = Cfg.load_config_from_file("config/resnet18-seq2seq.yml")
  vocab = Vocab(config['vocab'])

  img_path = 'vietocr/tests/image/test2.jpeg'
  img = Image.open(img_path)  
  # if show_img:
  #   plt.figure()
  #   plt.imshow(img) 
  #   plt.show()  # display it
  img = process_input(img, 32, 32, 1024) 

  cnn_session = onnxruntime.InferenceSession("cnn.onnx")
  encoder_session = onnxruntime.InferenceSession("encoder.onnx")
  decoder_session = onnxruntime.InferenceSession("decoder.onnx")

  img = np.array(img)
  session = (cnn_session, encoder_session, decoder_session)
  s = translate_onnx(img, session)
  s = s[0].tolist()
  s = vocab.decode(s)
  print("Result: ", s)

if __name__ == '__main__':
  # test_converter_onnx(show_img=False)
  run_onnx_converter()
