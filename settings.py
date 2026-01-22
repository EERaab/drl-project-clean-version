import matplotlib.pyplot as plt
import matplotlib.colors as colors
from IPython import display

import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from collections import namedtuple, deque
from itertools import count

#We define some aspects of the neural networks here so that our features can take them into account!
LOCAL_KERNEL_SIZE = 3

#The local window size is 2*(S + (kernel_size-1)/2)
LOCAL_WIDTH_RAD  = 6 + int((LOCAL_KERNEL_SIZE - 1)/2) #this has to do with the padding - we don't pad locally
LOCAL_HEIGHT_RAD = 6 + int((LOCAL_KERNEL_SIZE - 1)/2) #this has to do with the padding - we don't pad locally

GLOBAL_TILE_NUMBER = 9 #for dense worlds (which are 63x63)
#GLOBAL_KERNEL_SIZE = 15
#GLOBAL_STRIDE = 2*GLOBAL_KERNEL_SIZE #no overlap between global convolutions.

BATCH_SIZE = 64
MEMORY_SIZE = 10000

EPSILON = 0.1
LEARNING_RATE = 0.001
GAMMA = 0.99

WORLD_BOUNDARY_PADDING = max(LOCAL_WIDTH_RAD, LOCAL_HEIGHT_RAD)
WORLD_SEED_NR = 50

#Why is this funciton here? Who knows? Better leave it to avoid bugs downstream.
def trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')