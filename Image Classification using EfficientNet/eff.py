################################################################################
#
# LOGISTICS
#
#    <TO DO: Harsha Vardhini Vasu>
#    <TO DO: 2021495846>
#    <TO DO: this comment block is included in each file that is submitted:
#    eff.py, eff_se.py (if done) and eff_se_cond.py (if done)>
#
# FILE
#
#    eff.py | eff_se.py | eff_se_cond.py
#
# DESCRIPTION
#
#    Grade = eff.py grade (max 90) + eff_se.py grade (max 10) + eff_se_cond.py grade (max 10)
#
#    A PyTorch implementation of the network described in section 3 of
#    xNNs_Project_002_NetworksPaper.doc/pdf trained in Google Colaboratory using
#    a GPU instance (Runtime - Change runtime type - Hardware accelerator - GPU)
#
# INSTRUCTIONS
#
#    1. a. eff.py
#
#          Complete all <TO DO: ...> code portions of this file to design and
#          train the network in table 1 of the paper with the standard inverted
#          residual building block in fig 2a and report the results
#
#    1. b. eff_se.py
#
#          Starting from eff.py, create a SE block per fig 3 in the paper and
#          add it to the inverted residual block per fig 2b in the paper to
#          create a SE enhanced building block; design and train the network
#          in table 1 with the SE enhanced building block and report the same
#          results as in the standard building block case
#
#    1. c. eff_se_cond.py
#
#          Starting from eff_se.py, create a conditional conv operation per
#          fig 4 in the paper and replace the building block convolutions with
#          the conditional convolution operation to create a SE and conditional
#          convolution enhanced building block per fig 2c in the paper; design
#          and train the network in table 1 with the SE and conditional
#          convolution enhanced building block and report the same results as in
#          the standard building block case; it may be required to reduce the
#          batch size in this case if there is an out of memory error
#
#    2. Cut and paste the text output generated during training showing the per
#       epoch statistics (for all networks trained: standard, SE enhanced and
#       SE and conditional convolution enhanced)
#
#       <TO DO: cut and paste per epoch statistics here>
"""
Epoch   0 Time    692.5 lr = 0.002000 avg loss = 0.017957 accuracy =  1.20
Epoch   1 Time    689.0 lr = 0.041600 avg loss = 0.015924 accuracy = 13.88
Epoch   2 Time    687.1 lr = 0.081200 avg loss = 0.012965 accuracy = 25.42
Epoch   3 Time    687.5 lr = 0.120800 avg loss = 0.011372 accuracy = 32.26
Epoch   4 Time    686.3 lr = 0.160400 avg loss = 0.010325 accuracy = 37.36
Epoch   5 Time    687.3 lr = 0.200000 avg loss = 0.009586 accuracy = 36.72
Epoch   6 Time    688.1 lr = 0.199795 avg loss = 0.008948 accuracy = 43.98
Epoch   7 Time    689.9 lr = 0.199180 avg loss = 0.008440 accuracy = 43.50
Epoch   8 Time    689.4 lr = 0.198158 avg loss = 0.008061 accuracy = 45.56
Epoch   9 Time    689.3 lr = 0.196733 avg loss = 0.007778 accuracy = 50.20
Epoch  10 Time    689.0 lr = 0.194911 avg loss = 0.007494 accuracy = 52.18
Epoch  11 Time    689.0 lr = 0.192699 avg loss = 0.007269 accuracy = 53.16
Epoch  12 Time    688.9 lr = 0.190107 avg loss = 0.007067 accuracy = 53.34
Epoch  13 Time    689.2 lr = 0.187145 avg loss = 0.006902 accuracy = 57.36
Epoch  14 Time    689.3 lr = 0.183825 avg loss = 0.006739 accuracy = 57.70
Epoch  15 Time    689.0 lr = 0.180161 avg loss = 0.006603 accuracy = 57.88
Epoch  16 Time    689.2 lr = 0.176168 avg loss = 0.006468 accuracy = 57.42
Epoch  17 Time    689.2 lr = 0.171863 avg loss = 0.006342 accuracy = 56.34
Epoch  18 Time    688.8 lr = 0.167263 avg loss = 0.006230 accuracy = 57.88
Epoch  19 Time    689.2 lr = 0.162387 avg loss = 0.006121 accuracy = 59.96
Epoch  20 Time    688.4 lr = 0.157254 avg loss = 0.006000 accuracy = 58.56
Epoch  21 Time    688.9 lr = 0.151887 avg loss = 0.005936 accuracy = 62.96
Epoch  22 Time    688.6 lr = 0.146308 avg loss = 0.005838 accuracy = 62.94
Epoch  23 Time    686.6 lr = 0.140538 avg loss = 0.005725 accuracy = 61.12
Epoch  24 Time    686.8 lr = 0.134602 avg loss = 0.005642 accuracy = 61.92
Epoch  25 Time    685.5 lr = 0.128524 avg loss = 0.005555 accuracy = 63.48
Epoch  26 Time    685.6 lr = 0.122330 avg loss = 0.005481 accuracy = 63.86
Epoch  27 Time    685.4 lr = 0.116044 avg loss = 0.005384 accuracy = 63.24
Epoch  28 Time    685.4 lr = 0.109693 avg loss = 0.005296 accuracy = 64.90
Epoch  29 Time    685.1 lr = 0.103302 avg loss = 0.005205 accuracy = 66.64
Epoch  30 Time    685.0 lr = 0.096898 avg loss = 0.005068 accuracy = 63.60
Epoch  31 Time    684.7 lr = 0.090507 avg loss = 0.005002 accuracy = 65.30
Epoch  32 Time    685.1 lr = 0.084156 avg loss = 0.004919 accuracy = 65.64
Epoch  33 Time    685.0 lr = 0.077870 avg loss = 0.004826 accuracy = 66.58
Epoch  34 Time    684.5 lr = 0.071676 avg loss = 0.004728 accuracy = 66.64
Epoch  35 Time    685.4 lr = 0.065598 avg loss = 0.004647 accuracy = 65.30
Epoch  36 Time    685.6 lr = 0.059662 avg loss = 0.004541 accuracy = 66.16
Epoch  37 Time    685.6 lr = 0.053892 avg loss = 0.004444 accuracy = 67.50
Epoch  38 Time    685.4 lr = 0.048313 avg loss = 0.004362 accuracy = 67.62
Epoch  39 Time    685.2 lr = 0.042946 avg loss = 0.004247 accuracy = 68.68
Epoch  40 Time    685.4 lr = 0.037813 avg loss = 0.004183 accuracy = 68.72
Epoch  41 Time    685.2 lr = 0.032937 avg loss = 0.004057 accuracy = 67.90
Epoch  42 Time    685.1 lr = 0.028337 avg loss = 0.003962 accuracy = 68.02
Epoch  43 Time    684.6 lr = 0.024032 avg loss = 0.003876 accuracy = 70.66
Epoch  44 Time    684.7 lr = 0.020039 avg loss = 0.003827 accuracy = 69.52
Epoch  45 Time    684.5 lr = 0.016375 avg loss = 0.003721 accuracy = 70.22
Epoch  46 Time    684.7 lr = 0.013055 avg loss = 0.003667 accuracy = 70.10
Epoch  47 Time    685.1 lr = 0.010093 avg loss = 0.003577 accuracy = 70.18
Epoch  48 Time    684.9 lr = 0.007501 avg loss = 0.003513 accuracy = 70.78
Epoch  49 Time    684.7 lr = 0.005289 avg loss = 0.003483 accuracy = 70.48
Epoch  50 Time    684.7 lr = 0.003467 avg loss = 0.003439 accuracy = 71.10
Epoch  51 Time    684.5 lr = 0.002042 avg loss = 0.003412 accuracy = 71.18

"""
#
#    3. Submit eff.py, eff_se.py (if done) and eff_se_cond.py (if done) via
#       eLearning (no zip files, no Jupyter / iPython notebooks, ...) with this
#       comment block at the top and all code from the IMPORT comment block to
#       the end; so if you implement all 3, you will submit 3 Python files
#
# HELP
#
#    1. If you're looking for a reference for block and network design, see
#       https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/blob/master/Tests/202008/xNNs_Project_002_Networks.py
#       which implemented a RegNetX style block and network; while the block and
#       network is different, that code should help with thinking about how to
#       organize this code
#
################################################################################


# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EoR8xxXNHOPFKKBxUlDWPAlc1n60K_MZ

IMPORT
"""

# torch
import torch
import torch.nn       as     nn
import torch.optim    as     optim
from   torch.autograd import Function

# torch utils
import torchvision
import torchvision.transforms as transforms

# additional libraries
import os
import urllib.request
import zipfile
import time         
import math
import numpy             as np
import matplotlib.pyplot as plt

"""PARAMETERS"""

# data
DATA_DIR_1        = 'data'
DATA_DIR_2        = 'data/imagenet64'
DATA_DIR_TRAIN    = 'data/imagenet64/train'
DATA_DIR_TEST     = 'data/imagenet64/val'
DATA_FILE_TRAIN_1 = 'Train1.zip'
DATA_FILE_TRAIN_2 = 'Train2.zip'
DATA_FILE_TRAIN_3 = 'Train3.zip'
DATA_FILE_TRAIN_4 = 'Train4.zip'
DATA_FILE_TRAIN_5 = 'Train5.zip'
DATA_FILE_TEST_1  = 'Val1.zip'
DATA_URL_TRAIN_1  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train1.zip'
DATA_URL_TRAIN_2  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train2.zip'
DATA_URL_TRAIN_3  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train3.zip'
DATA_URL_TRAIN_4  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train4.zip'
DATA_URL_TRAIN_5  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train5.zip'
DATA_URL_TEST_1   = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Val1.zip'
DATA_BATCH_SIZE   = 256
DATA_NUM_WORKERS  = 4
DATA_NUM_CHANNELS = 3
DATA_NUM_CLASSES  = 100
DATA_RESIZE       = 64
DATA_CROP         = 56
DATA_MEAN         = (0.485, 0.456, 0.406)
DATA_STD_DEV      = (0.229, 0.224, 0.225)

# model
# <TO DO: your code goes here>
MODEL_LEVEL_1_BLOCKS   = 0
MODEL_LEVEL_2_BLOCKS   = 0
MODEL_LEVEL_3_BLOCKS   = 1
MODEL_LEVEL_4_BLOCKS   = 2
MODEL_LEVEL_5_BLOCKS   = 3
MODEL_LEVEL_6_BLOCKS   = 4
MODEL_LEVEL_1_CHANNELS = 16
MODEL_LEVEL_2_CHANNELS = 16
MODEL_LEVEL_3_CHANNELS = 24
MODEL_LEVEL_4_CHANNELS = 40
MODEL_LEVEL_5_CHANNELS = 80
MODEL_LEVEL_6_CHANNELS = 160
HEAD_CONV_CHANNELS = 320
HEAD_AVG_POOL_CHANNELS = 1280 

# training
TRAIN_LR_MAX              = 0.2
TRAIN_LR_INIT_SCALE       = 0.01
TRAIN_LR_FINAL_SCALE      = 0.001
TRAIN_LR_INIT_EPOCHS      = 5
TRAIN_LR_FINAL_EPOCHS     = 50 # 100
TRAIN_NUM_EPOCHS          = TRAIN_LR_INIT_EPOCHS + TRAIN_LR_FINAL_EPOCHS
TRAIN_LR_INIT             = TRAIN_LR_MAX*TRAIN_LR_INIT_SCALE
TRAIN_LR_FINAL            = TRAIN_LR_MAX*TRAIN_LR_FINAL_SCALE
TRAIN_INTRA_EPOCH_DISPLAY = 10000

# file
FILE_NAME_CHECK      = 'EffNetStyleCheck.pt'
FILE_NAME_BEST       = 'EffNetStyleBest.pt'
FILE_SAVE            = True
FILE_LOAD            = False
FILE_EXTEND_TRAINING = False
FILE_NEW_OPTIMIZER   = False

"""DATA"""

# create a local directory structure for data storage
if (os.path.exists(DATA_DIR_1) == False):
    os.mkdir(DATA_DIR_1)
if (os.path.exists(DATA_DIR_2) == False):
    os.mkdir(DATA_DIR_2)
if (os.path.exists(DATA_DIR_TRAIN) == False):
    os.mkdir(DATA_DIR_TRAIN)
if (os.path.exists(DATA_DIR_TEST) == False):
    os.mkdir(DATA_DIR_TEST)

# download data
if (os.path.exists(DATA_FILE_TRAIN_1) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_1, DATA_FILE_TRAIN_1)
if (os.path.exists(DATA_FILE_TRAIN_2) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_2, DATA_FILE_TRAIN_2)
if (os.path.exists(DATA_FILE_TRAIN_3) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_3, DATA_FILE_TRAIN_3)
if (os.path.exists(DATA_FILE_TRAIN_4) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_4, DATA_FILE_TRAIN_4)
if (os.path.exists(DATA_FILE_TRAIN_5) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_5, DATA_FILE_TRAIN_5)
if (os.path.exists(DATA_FILE_TEST_1) == False):
    urllib.request.urlretrieve(DATA_URL_TEST_1, DATA_FILE_TEST_1)

# extract data
with zipfile.ZipFile(DATA_FILE_TRAIN_1, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_2, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_3, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_4, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_5, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TEST_1, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TEST)

# transforms
transform_train = transforms.Compose([transforms.RandomResizedCrop(DATA_CROP), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])
transform_test  = transforms.Compose([transforms.Resize(DATA_RESIZE), transforms.CenterCrop(DATA_CROP), transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])

# data sets
dataset_train = torchvision.datasets.ImageFolder(DATA_DIR_TRAIN, transform=transform_train)
dataset_test  = torchvision.datasets.ImageFolder(DATA_DIR_TEST,  transform=transform_test)

# data loader
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=DATA_BATCH_SIZE, shuffle=True)
dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=DATA_BATCH_SIZE, shuffle=False)

"""NETWORK BUILDING BLOCK"""

# inverted residual block
class InvResBlock(nn.Module):
    
    # initialization
    def __init__(self, Ni, Ne, No, F, S, ID):

        self.ID = ID

        # parent initialization
        super(InvResBlock, self).__init__()

        # create all of the operators for the inverted residual block in fig 2a
        # of the paper; note that parameter names were chosen to match the paper
        # <TO DO: your code goes here>
        
        P = np.floor(F/2).astype(int)
        self.conv1 = nn.Conv2d(Ni, Ne, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.bn1   = nn.BatchNorm2d(Ne, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(Ne, Ne, (F, F), stride=(S, S), padding=(P, P), dilation=(1, 1), groups=Ne, bias=False, padding_mode='zeros')
        self.bn2   = nn.BatchNorm2d(Ne, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(Ne, No, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros')
        self.bn3   = nn.BatchNorm2d(No, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU()

    # forward path
    def forward(self, x):

        # map input x to output y for the inverted residual block in fig 2a of
        # the paper via connecting the operators defined in the initialization
        # and return output y
        # <TO DO: your code goes here>

        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu1(res)
        res = self.conv2(res)
        res = self.bn2(res)
        res = self.relu2(res)
        res = self.conv3(res)
        res = self.bn3(res)

        if self.ID == True:
          y = self.relu3(x+res)
        else:
          y = self.relu3(res)

        # return
        return y

"""NETWORK"""

# define
class Model(nn.Module):

    # initialization
    # add necessary parameters to the init function to create the model defined
    # in table 1 of the paper
    def __init__(self, 
                 data_num_channels,
                 model_level_1_blocks, model_level_1_channels,
                 model_level_2_blocks, model_level_2_channels, 
                 model_level_3_blocks, model_level_3_channels,
                 model_level_4_blocks, model_level_4_channels,
                 model_level_5_blocks, model_level_5_channels,
                 model_level_6_blocks, model_level_6_channels,
                 head_conv_channels, head_avg_pool_channels, 
                 data_num_classes): # <TO DO: your code goes here> inside the parenthesis

        # parent initialization
        super(Model, self).__init__()

        # create all of the operators for the network defined in table 1 of the
        # paper using a combination of Python, standard PyTorch operators and
        # the previously defined InvResBlock class
        # <TO DO: your code goes here>

        # stride
        stride1 = 1 
        stride2 = 1 
        stride3 = 1
        stride4 = 2
        stride5 = 2
        stride6 = 2
        stride7 = 1

        # encoder level 1 - Tail
        self.enc_1 = nn.ModuleList()
        self.enc_1.append(nn.Conv2d(data_num_channels, model_level_1_channels, (3, 3), stride=(stride1, stride1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))
        self.enc_1.append(nn.BatchNorm2d(model_level_1_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.enc_1.append(nn.ReLU())

        # encoder level 2 - Bloack 1
        self.enc_2 = nn.ModuleList()
        self.enc_2.append(InvResBlock(model_level_1_channels, 4 * model_level_1_channels, model_level_2_channels, 3, stride2, True))

        # encoder level 3 - Block 2
        self.enc_3 = nn.ModuleList()
        self.enc_3.append(InvResBlock(model_level_2_channels, 4 * model_level_2_channels, model_level_3_channels, 3, stride3, False))
        for n in range(model_level_3_blocks):
            self.enc_3.append(InvResBlock(model_level_3_channels, 4 * model_level_3_channels, model_level_3_channels, 3, 1, True))

        # encoder level 4 - Block 3
        self.enc_4 = nn.ModuleList()
        self.enc_4.append(InvResBlock(model_level_3_channels, 4 * model_level_3_channels, model_level_4_channels, 3, stride4, False))
        for n in range(model_level_4_blocks):
            self.enc_4.append(InvResBlock(model_level_4_channels, 4 * model_level_4_channels, model_level_4_channels, 3, 1, True))

        # encoder level 5 - Block 4
        self.enc_5 = nn.ModuleList()
        self.enc_5.append(InvResBlock(model_level_4_channels, 4 * model_level_4_channels, model_level_5_channels, 3, stride5, False))
        for n in range(model_level_5_blocks):
            self.enc_5.append(InvResBlock(model_level_5_channels, 4 * model_level_5_channels, model_level_5_channels, 3, 1, True))

        # encoder level 6 - Block 5
        self.enc_6 = nn.ModuleList()
        self.enc_6.append(InvResBlock(model_level_5_channels, 4 * model_level_5_channels, model_level_6_channels, 3, stride5, False))
        for n in range(model_level_6_blocks):
            self.enc_6.append(InvResBlock(model_level_6_channels, 4 * model_level_6_channels, model_level_6_channels, 3, 1, True))
        self.enc_6.append(InvResBlock(model_level_6_channels, 4 * model_level_6_channels, head_conv_channels, 3, stride6, False))
        
        # decoder
        self.dec = nn.ModuleList()
        self.dec.append(nn.Conv2d(head_conv_channels, head_avg_pool_channels, (1, 1), stride=(stride7, stride7), padding=(1, 1), dilation=(1, 1), groups=1, bias=False, padding_mode='zeros'))
        self.dec.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.dec.append(nn.Flatten())
        self.dec.append(nn.Linear(head_avg_pool_channels, data_num_classes, bias=True))

    # forward path
    def forward(self, x):

        # map input x to output y for the network defined in table 1 of the
        # paper via connecting the operators defined in the initialization
        # and return output y
        # <TO DO: your code goes here>

        # encoder level 1
        for layer in self.enc_1:
            x = layer(x)


        # encoder level 2
        for layer in self.enc_2:
            x = layer(x)

        # encoder level 3
        for layer in self.enc_3:
            x = layer(x)

        # encoder level 4
        for layer in self.enc_4:
            x = layer(x)

        # encoder level 5
        for layer in self.enc_5:
            x = layer(x)
        
        # encoder level 6
        for layer in self.enc_6:
            x = layer(x)

        # decoder
        for layer in self.dec:
            x = layer(x)
        y = x
        # return
        return y

# create
# add necessary parameters to the init function to create the model defined
# in table 1 of the paper
model = Model(DATA_NUM_CHANNELS,
              MODEL_LEVEL_1_BLOCKS, MODEL_LEVEL_1_CHANNELS, 
              MODEL_LEVEL_2_BLOCKS, MODEL_LEVEL_2_CHANNELS, 
              MODEL_LEVEL_3_BLOCKS, MODEL_LEVEL_3_CHANNELS, 
              MODEL_LEVEL_4_BLOCKS, MODEL_LEVEL_4_CHANNELS, 
              MODEL_LEVEL_5_BLOCKS, MODEL_LEVEL_5_CHANNELS, 
              MODEL_LEVEL_6_BLOCKS, MODEL_LEVEL_6_CHANNELS, 
              HEAD_CONV_CHANNELS, HEAD_AVG_POOL_CHANNELS,
              DATA_NUM_CLASSES) # <TO DO: your code goes here> inside the parenthesis

# enable data parallelization for multi GPU systems
if (torch.cuda.device_count() > 1):
    model = nn.DataParallel(model)
print('Using {0:d} GPU(s)'.format(torch.cuda.device_count()), flush=True)

"""ERROR AND OPTIMIZER"""

# error (softmax cross entropy)
criterion = nn.CrossEntropyLoss()

# learning rate schedule
def lr_schedule(epoch):

    # linear warmup followed by 1/2 wave cosine decay
    if epoch < TRAIN_LR_INIT_EPOCHS:
        lr = (TRAIN_LR_MAX - TRAIN_LR_INIT)*(float(epoch)/TRAIN_LR_INIT_EPOCHS) + TRAIN_LR_INIT
    else:
        lr = TRAIN_LR_FINAL + 0.5*(TRAIN_LR_MAX - TRAIN_LR_FINAL)*(1.0 + math.cos(((float(epoch) - TRAIN_LR_INIT_EPOCHS)/(TRAIN_LR_FINAL_EPOCHS - 1.0))*math.pi))

    return lr

# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, dampening=0.0, weight_decay=5e-5, nesterov=True)

"""TRAINING"""

# start epoch
start_epoch = 0

# specify the device as the GPU if present with fallback to the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transfer the network to the device
model.to(device)

# load the last checkpoint
if (FILE_LOAD == True):
    checkpoint = torch.load(FILE_NAME_CHECK)
    model.load_state_dict(checkpoint['model_state_dict'])
    if (FILE_NEW_OPTIMIZER == False):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if (FILE_EXTEND_TRAINING == False):
        start_epoch = checkpoint['epoch'] + 1

# initialize the epoch
accuracy_best      = 0
start_time_display = time.time()
start_time_epoch   = time.time()

# cycle through the epochs
for epoch in range(start_epoch, TRAIN_NUM_EPOCHS):

    # initialize epoch training
    model.train()
    training_loss = 0.0
    num_batches   = 0
    num_display   = 0

    # set the learning rate for the epoch
    for g in optimizer.param_groups:
        g['lr'] = lr_schedule(epoch)

    # cycle through the training data set
    for data in dataloader_train:

        # extract a batch of data and move it to the appropriate device
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass, loss, backward pass and weight update
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # update statistics
        training_loss = training_loss + loss.item()
        num_batches   = num_batches + 1
        num_display   = num_display + DATA_BATCH_SIZE

        # display intra epoch results
        if (num_display > TRAIN_INTRA_EPOCH_DISPLAY):
            num_display          = 0
            elapsed_time_display = time.time() - start_time_display
            start_time_display   = time.time()
            print('Epoch {0:3d} Time {1:8.1f} lr = {2:8.6f} avg loss = {3:8.6f}'.format(epoch, elapsed_time_display, lr_schedule(epoch), (training_loss / num_batches) / DATA_BATCH_SIZE), flush=True)

    # initialize epoch testing
    model.eval()
    test_correct = 0
    test_total   = 0

    # no weight update / no gradient needed
    with torch.no_grad():

        # cycle through the testing data set
        for data in dataloader_test:

            # extract a batch of data and move it to the appropriate device
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass and prediction
            outputs      = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # update test set statistics
            test_total   = test_total + labels.size(0)
            test_correct = test_correct + (predicted == labels).sum().item()

    # epoch statistics
    elapsed_time_epoch = time.time() - start_time_epoch
    start_time_epoch   = time.time()
    print('Epoch {0:3d} Time {1:8.1f} lr = {2:8.6f} avg loss = {3:8.6f} accuracy = {4:5.2f}'.format(epoch, elapsed_time_epoch, lr_schedule(epoch), (training_loss/num_batches)/DATA_BATCH_SIZE, (100.0*test_correct/test_total)), flush=True)

    # save a checkpoint
    if (FILE_SAVE == True):
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, FILE_NAME_CHECK)

    # save the best model
    accuracy_epoch = 100.0 * test_correct / test_total
    if ((FILE_SAVE == True) and (accuracy_epoch >= accuracy_best)):
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, FILE_NAME_BEST)