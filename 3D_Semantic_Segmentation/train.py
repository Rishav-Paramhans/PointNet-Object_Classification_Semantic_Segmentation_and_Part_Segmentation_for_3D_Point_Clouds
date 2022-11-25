#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataloader import S3DIS
from model import STN3d,STNkd,PointNetfeature,PointNetSemanticSegmenter,feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd


# In[2]:



dataset = S3DIS(train = True, validation = False,randomize_data = False)
test_dataset = S3DIS(train = False, validation = False,randomize_data = False)
validation_dataset = S3DIS(train = True, validation = True,randomize_data = False)

dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=0)
testdataloader = torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=True,num_workers=0)
validationloader = torch.utils.data.DataLoader(validation_dataset,batch_size=32,shuffle=True,num_workers=0)

#print(len(dataset), len(validation_dataset), len(test_dataset))


# In[3]:


#num_class = 13
model =  PointNetSemanticSegmenter(k=13, feature_transform=True) #k= num_classes
FILE = "model.pth"
torch.save(model.state_dict(),FILE)


# In[4]:


semantic_segmenter = PointNetSemanticSegmenter(k=13, feature_transform=True)   #k= num_classes
semantic_segmenter.load_state_dict(torch.load(FILE))
semantic_segmenter.cuda()


# In[5]:


optimizer = optim.Adam(semantic_segmenter.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


# In[6]:


try:
    os.makedirs("Trained_Sem_Seg_Model")
except OSError:
    pass


# In[7]:


validation_df = pd.DataFrame(columns=["epoch", "i", "num_batch", "validation_loss", "accuracy"])
training_df = pd.DataFrame(columns=["epoch", "i", "num_batch", "validation_loss", "accuracy"])


# In[8]:


batchSize = 32
num_batch = len(dataset) /batchSize
num_epoch = 120
for epoch in range(num_epoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points,target = data
        points = points.transpose(2, 1)
        #print(points.size())
        #print(target.size())
        points,target = points.cuda(),target.cuda()
        semantic_segmenter = semantic_segmenter.train()
        
        #forwardpass
        pred, trans, trans_feat = semantic_segmenter(points)
        pred = pred.view(-1, 13)
        target = target.view(-1, 1)[:, 0]
        #print(pred.size(), target.size())
        #print('pred',pred)
        #print('target',target)
        loss = F.nll_loss(pred, target)
        if semantic_segmenter.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
       
        
        
        #Backwardpass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(batchSize * 4096)))
        training_df= training_df.append({"epoch": epoch, "i":i, "num_batch":num_batch, "train_loss":loss.item(), "accuracy":correct.item()/float(batchSize * 4096)},ignore_index=True)
        if i % 10 == 0:
            j, data = next(enumerate(validationloader, 0))
            points,target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            semantic_segmenter = semantic_segmenter.eval()
            pred, _, _ = semantic_segmenter(points)
            pred = pred.view(-1, 13)
            target = target.view(-1, 1)[:, 0]
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, 'validation', loss.item(), correct.item()/float(batchSize * 4096)))
            validation_df = validation_df.append({"epoch": epoch, "i":i, "num_batch":num_batch, "validation_loss":loss.item(), "accuracy":correct.item()/float(batchSize * 4096)},ignore_index=True)
    torch.save(semantic_segmenter.state_dict(), '%s\semantic_seg_model_%d.pth' %("Trained_Sem_Seg_Model",epoch))
    training_df.to_csv('training.csv')    
    validation_df.to_csv('validation.csv') 


# In[ ]:





# In[26]:




