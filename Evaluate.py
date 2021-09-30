# Evaluate model
import torchvision.models as models
import torch
import copy
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import json
import cv2

logs_dir="logs/"
if not os.path.exists(logs_dir): os.mkdir(logs_dir)
#TrainJson=r'archive/ss304/test/test.json'
#TrainFolder=r'archive/ss304/test/'

TrainJson=r'archive/ss304/test/test.json'
TrainFolder=r'archive/ss304/test/'
train_series = pd.read_json(TrainJson,typ='series')
train_dataframe = pd.DataFrame(train_series)
train_dataframe.head()
train_dataframe = train_dataframe.reset_index()

train_dataframe.columns = ['filename','category']
train_dataframe.head()
Trained_model_path="logs/3000.torch"

numcats=6
h=700
w=1280
#--------------Create file list------------------------------------------------------------
filesbycat={}
batchSize=1
for i in range(numcats):
    filesbycat[i]=[]


for  f in range(len(train_dataframe['filename'])):
    # print(train_dataframe['filename'][f],train_dataframe['category'][f])
     filesbycat[train_dataframe['category'][f]].append(train_dataframe['filename'][f])
for ct in filesbycat:
    print(ct, len(filesbycat[ct]))

#----------------Create net class--------------------------------------------------------------------
class NetModel(nn.Module):# 
######################Load main net (resnet 50) class############################################################################################################
        def __init__(self,NumClasses): # Load pretrained encoder and prepare net layers
            super(NetModel, self).__init__()
# ---------------Load pretrained torchvision resnet (need to be connected to the internet to download model in the first time)----------------------------------------------------------
            self.Net = models.resnext50_32x4d(pretrained=False)
#----------------Change Final prediction fully connected layer from imagnet 1000 classes to coco 80 classes------------------------------------------------------------------------------------------
            self.Net.fc=nn.Linear(2048, NumClasses)
###############################################Run prediction inference using the net ###########################################################################################################
        def forward(self,Images,EvalMode=False,UseGPU=True):
#------------------------------- Convert from numpy to pytorch-------------------------------------------------------
                InpImages = torch.autograd.Variable(torch.from_numpy(Images), requires_grad=False).transpose(2,3).transpose(1, 2).type(torch.FloatTensor)
                if UseGPU == True: # Convert to GPU
                    InpImages = InpImages.cuda()
# -------------------------Normalize image-------------------------------------------------------------------------------------------------------
                RGBMean = [123.68, 116.779, 103.939]
                RGBStd = [65, 65, 65]
                for i in range(len(RGBMean)): InpImages[:, i, :, :]=(InpImages[:, i, :, :]-RGBMean[i])/RGBStd[i] # Normalize image by std and mean
#----------------Run prediction-----------------------------------------------                
                x=self.Net(InpImages)
                ProbVec = F.softmax(x,dim=1) # Probability vector for all classes
                Prob,Pred=ProbVec.max(dim=1) # Top predicted class and probability
                return ProbVec,Pred
#########################Load net############################################################################################

Net=NetModel(numcats)

Net.load_state_dict(torch.load(Trained_model_path))
Net.cuda()
Net=Net.eval()
AVGLoss=0
#-------------------Statitics data---------------------------------------------------------------
SumPrdClass=np.zeros([numcats],np.float32)
SumGTClass=np.zeros([numcats],np.float32)
SumTPClass=np.zeros([numcats],np.float32)
itrClass=np.zeros([numcats],np.int32)
TP=0
TotalPred=0
#############################################################################################################################
for itr in range(100000):

#*********************************Load data*************************************************************************************************
    Images=np.zeros([batchSize,h,w,3],np.float32)
    GTLabels=np.zeros([batchSize],np.float32)
    for i in range(batchSize):
        ct=itr%numcats
        f=itrClass[ct]

        if f>=len(filesbycat[ct]): continue
        im=cv2.imread(TrainFolder+"//"+filesbycat[ct][f])
        itrClass[ct] += 1

        Images[i]=im
        GTLabels[i]=ct
    if itrClass[ct] >= len(filesbycat[ct]): continue

#**************************Run Trainin cycle***************************************************************************************
    with torch.no_grad():
             Prob, Lb=Net.forward(Images) # Run net inference and get prediction

# --------------Evaluate------------------------------------------------------------------------------------------------------------------------------------------
    PrdLB=Lb.data.cpu().numpy()
    TotalPred+=batchSize
    TP+=(PrdLB==GTLabels).sum()

    
    for i in range(numcats):
        SumPrdClass[i] +=(PrdLB==i).sum()
        SumGTClass[i] += (GTLabels == i).sum()
        SumTPClass[i] += ((PrdLB==GTLabels)*(GTLabels == i)).sum()
#-----------Display-----------------------------------------------------------------------
    print()
    print("-----------",itr,"-------------------------------------------")
    RecallAv=0
    for i in range(numcats):
              RecallAv+= SumTPClass[i] / SumGTClass[i]
              print("Class:",i,") Recall:", SumTPClass[i] / SumGTClass[i], "Precision:", SumTPClass[i] / SumPrdClass[i], "Sum GT ", SumGTClass[i],  " Sum Predicted ", SumPrdClass[i])
    print("Average Recall (accuracy Class Average)",RecallAv/numcats)
    print("Accuracy (no class blance)", TP / TotalPred, "Total number of prediction: ", TotalPred)

        
        
