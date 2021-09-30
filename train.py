# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
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
Learning_Rate=1e-5 #Learning rate for Adam Optimizer
learning_rate_decay=0.999999#
Weight_Decay=1e-4# Weight for the weight decay loss function
logs_dir="logs/"
if not os.path.exists(logs_dir): os.mkdir(logs_dir)
TrainJson=r'archive/ss304/train/train.json'
TrainFolder=r'archive/ss304/train/'
train_series = pd.read_json(TrainJson,typ='series')
train_dataframe = pd.DataFrame(train_series)
train_dataframe.head()
train_dataframe = train_dataframe.reset_index()

train_dataframe.columns = ['filename','category']
train_dataframe.head()
Trained_model_path=""
TrainLossTxtFile="Trainloss.txt"
numcats=6
h=700
w=1280
#--------------Create file list------------------------------------------------------------
filesbycat={}
batchSize=10
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
            self.Net = models.resnext50_32x4d(pretrained=True)
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
if Trained_model_path!="":
    Net.load_state_dict(torch.load(Trained_model_path))
Net.cuda()
#optimizer=torch.optim.SGD(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay,momentum=0.5)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay)
#--------------------------- Create files for saving loss----------------------------------------------------------------------------------------------------------
f = open(TrainLossTxtFile, "a")
f.write("Iteration\tloss\t Learning Rate="+str(Learning_Rate))
f.close()
AVGLoss=0
TP=0
TotalPred=0
#############################################################################################################################
for itr in range(10000000):

#*********************************Load data*************************************************************************************************
    Images=np.zeros([batchSize,h,w,3],np.float32)
    HotLabels=np.zeros([batchSize,numcats],np.float32)
    StandartLabels=np.zeros([batchSize],np.float32)
    for i in range(batchSize):
       ct=np.random.randint(0,numcats)
       f=np.random.randint(0,len(filesbycat[ct]))
       im=cv2.imread(TrainFolder+"//"+filesbycat[ct][f])
       if np.random.rand()<0.5: im=np.fliplr(im)
       if np.random.rand() < 0.5: im = np.flipud(im)

       Images[i]=im
       HotLabels[i,ct]=1
       StandartLabels[i]=ct
#**************************************************************************************************************************
    # Images[:,:,:,1]*=SegmentMask
    # for ii in range(Labels.shape[0]):
    #     print(Reader.CatNames[Labels[ii]])
    #     misc.imshow(Images[ii])
#**************************Run Trainin cycle***************************************************************************************
    Prob, Lb=Net.forward(Images) # Run net inference and get prediction
    Net.zero_grad()
    OneHotLabels=torch.autograd.Variable(torch.from_numpy(HotLabels).cuda(), requires_grad=False)
    Loss = -torch.mean((OneHotLabels * torch.log(Prob + 0.0000001)))  # Calculate cross entropy loss
    if AVGLoss==0:  AVGLoss=float(Loss.data.cpu().numpy()) #Caclculate average loss for display
    else: AVGLoss=AVGLoss*0.999+0.001*float(Loss.data.cpu().numpy())
    Loss.backward() # Backpropogate loss
    optimizer.step() # Apply gradient decend change weight
    torch.cuda.empty_cache()
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
    TotalPred+=batchSize
    TP+=(Lb.data.cpu().numpy()==StandartLabels).sum()
    print("Accuracy",TP/TotalPred)
    if itr % 1000 == 0 and itr>0:
        print("Saving Model to file in "+logs_dir)
        torch.save(Net.state_dict(), logs_dir+ "/" + str(itr) + ".torch")
        print("model saved")
#......................Write and display train loss..........................................................................
    if itr % 10==0: # Display train loss
        print("Step "+str(itr)+" Train Loss="+str(float(Loss.data.cpu().numpy()))+" Runnig Average Loss="+str(AVGLoss))
        #Write train loss to file
        with open(TrainLossTxtFile, "a") as f:
            f.write("\n"+str(itr)+"\t"+str(float(Loss.data.cpu().numpy()))+"\t"+str(AVGLoss))
            f.close()