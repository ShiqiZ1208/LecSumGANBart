import argparse
from Train import train_model, model_predict
from Evaluation import Evaluated
from bart_train import Btrain_model
import os



parser = argparse.ArgumentParser()
# Option run or train
parser.add_argument("-o", "--options",)

# training option

# Option number of epoch for train
parser.add_argument("-e", "--num_epochs",)
# number of batch size
parser.add_argument("-b","--batch_size")
# save the model
parser.add_argument("-save","--is_save")
# if load model
parser.add_argument("-l","--is_load")
# load the model
parser.add_argument("-g","--load_pathG")
parser.add_argument("-d","--load_pathD")

# Predict option

#compare fine-tuned trained model with pretrained model
parser.add_argument("-c", "--is_compare")
#input text or article
parser.add_argument("-i", "--inputs_path")

#evaluation
parser.add_argument("-mode", "--model")
parser.add_argument("-base", "--is_baseline")


# load all the arguments
args = parser.parse_args()
options = args.options
num_epochs = args.num_epochs
batch_size = args.batch_size
is_save = args.is_save
is_load = args.is_load
load_pathG = args.load_pathG
load_pathD = args.load_pathD
modes = args.model

is_compare = args.is_compare
t_input = args.inputs_path
is_base = args.is_baseline

if options.lower() == "train":
  num_epochs = int(num_epochs)
  batch_size = int(batch_size)
  if isinstance(int(num_epochs), int) == False:
      raise Exception("Type error: num_epochs must be int")

  if isinstance(int(batch_size), int) == False:
      raise Exception("Type error: batch_size must be int")

  if isinstance(is_save, str):
      if is_save.lower() == "true":
        is_save = True
      elif is_save.lower() == "false":
        is_save = False
  else:
      raise Exception("Type error: need type string for is_save")

  if isinstance(is_load, str):
      if is_load.lower() == "true":
        is_load = True
      elif is_load.lower() == "false":
        is_load = False
  else:
      raise Exception("Type error: need type string for is_load")

  if is_load == True:
    if os.path.exists(load_pathG):
        print(f"Load the BART model from {load_pathG}.")
    else:
        print(f"The file {load_pathG} does not exist.")
    if os.path.exists(load_pathD):
        print(f"Load the BERT model from {load_pathD}.")
    else:
        print(f"The file {load_pathD} does not exist.")

elif options.lower() == "predict":
  if os.path.exists(load_pathG):
          print(f"Load the BART model from {load_pathG}.")
  else:
          print(f"The file {load_pathG} does not exist.")
          raise Exception("input error: model file not exist")

  if os.path.exists(t_input):
          print(f"Load the input text from {t_input}.")
  else:
          print(f"The file {t_input} does not exist.")
          raise Exception("input error: input file not exist")
if is_base != None:          
    if is_base.lower() == "true":
      is_baseline = True
    elif is_base.lower() == "false":
      is_baseline = False
    else:
      pass
else:
  pass

if isinstance(options, str):
    if options.lower() == "train":
        if modes == "GAN":
          print("train_GAN")
          train_model(num_epochs, batch_size, is_save, is_load, load_pathG, load_pathD, BART_only=False)
        if modes == "BART":
          Btrain_model(num_epochs, batch_size)
    elif options.lower() == "predict":
        model_predict(t_input, load_pathG)
    elif options.lower() == "evaluate":
        Evaluated(load_pathG)
else:
    raise Exception("Type error: need type string")




