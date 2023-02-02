import argparse
from PIL import Image
import os
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.tool.utils import compute_accuracy


def replace(s, ch):
    new_str = []
    l = len(s)
     
    for i in range(len(s)):
        if (s[i] == ch and i != (l-1) and
           i != 0 and s[i + 1] != ch and s[i-1] != ch):
            new_str.append(s[i])
             
        elif s[i] == ch:
            if ((i != (l-1) and s[i + 1] == ch) and
               (i != 0 and s[i-1] != ch)):
                new_str.append(s[i])
                 
        else:
            new_str.append(s[i])
         
    return ("".join(i for i in new_str))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='foo help')
    parser.add_argument('--annotation', required=True, help='foo help')
    parser.add_argument('--config', required=True, help='foo help')
    parser.add_argument('--res_dir', required=True, help='foo help')
   
    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)
    print(config)
    os.makedirs(args.res_dir+'/incorrect',exist_ok=True)
    os.makedirs(args.res_dir+'/correct',exist_ok=True)
    detector = Predictor(config)
    count = 0
    count_correct = 0
    f = open(args.annotation,'r')
    lines = f.readlines()
    #actual_sents = []
    #pred_sents = []
    for i,line in enumerate(lines):
    #   print("i = {}, line = {}".format(i, line))
      line_content = line[:-1].split("\t")
      label = line_content[1]
      img_path = os.path.join(args.root,line_content[0])
      img = Image.open(img_path)
      s, prob, char_prob = detector.predict(img,return_prob=True)
      # REGULATION
      #if all(x.isnumeric() or x.isspace() or x == '.' for x in s):
      #  s = s.replace(" ","")
      #if all(x.isalpha() or x.isspace() for x in s):
      #  s = s.replace(" ","")
      #s = s.replace("<","(")
      #s = s.replace(">",")")
      
      #s = replace(s," ")
      #label = replace(label," ")


      save_name = '{}_'.format(i)+s+'vs'+label+'.jpg'
      save_name = save_name.replace("/","?")
      if s != label:
        print(save_name)
        img.save(os.path.join(args.res_dir+'/incorrect',save_name))
        count+=1
      else:
        img.save(os.path.join(args.res_dir+'/correct',save_name))      
        count_correct+=1  
    #acc_full_seq = compute_accuracy(actual_sents, pred_sents, mode='full_sequence')
    print('incorrect_rate:', count/len(lines))
    print('correct_rate:', count_correct/len(lines))
    #print(acc_full_seq)
    f.close()

if __name__ == '__main__':
    main()
