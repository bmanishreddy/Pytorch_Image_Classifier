import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import torchvision.models as models

import argparse

#import Train
import predict

ap = argparse.ArgumentParser(description='predict.py')
# Command Line ardguments



ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
#ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./flowers/test/10/image_07090")
ap.add_argument('--top_kk', dest="top_kk", action="store", default=1)
#ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

pa = ap.parse_args()
dirpath = pa.data_dir
path = pa.save_dir
top_k = pa.top_kk



model =predict.load_checkpoint_cpu()

#img_test_o = process_image(image_graph)
#img_test_o = process_image(image_graph)
predict.PredictProb(path,model,top_k)