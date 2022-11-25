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
from dataloader import ModelNet40Data
from model import STN3d, STNkd, PointNetfeature,PointNetClassifier,feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd


# In[2]:


dataset = ModelNet40Data(train = True, validation = False, num_points = 2048, randomize_data = False, use_normals = False)
test_dataset = ModelNet40Data(train = False, validation = False, num_points = 2048, randomize_data = False, use_normals = False)
validation_dataset = ModelNet40Data(train = True, validation = True, num_points = 2048, randomize_data = False, use_normals = False)

dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=0)
testdataloader = torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=True,num_workers=0)
validationloader = torch.utils.data.DataLoader(validation_dataset,batch_size=32,shuffle=True,num_workers=0)
#print(len(dataset), len(validation_dataset), len(test_dataset))


# In[3]:


model =  PointNetClassifier(k=40, feature_transform=True)
FILE = "model.pth"
torch.save(model.state_dict(),FILE)
TRAINED_FILE = "trained_model.pth"


# In[4]:


classifier = PointNetClassifier(k=40, feature_transform=True)
classifier.load_state_dict(torch.load(FILE))


# In[5]:


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()
#print(optimizer.state_dict())


# In[6]:


validation_df = pd.DataFrame(columns=["epoch", "i", "num_batch", "validation_loss", "accuracy"])
training_df = pd.DataFrame(columns=["epoch", "i", "num_batch", "validation_loss", "accuracy"])


# In[7]:


try:
    os.makedirs("Trained_Model")
except OSError:
    pass


# In[8]:


batchSize = 32
num_batch = len(dataset) / batchSize
num_epoch = 120
for epoch in range(num_epoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        #points = points.transpose(2, 1)
        points = torch.permute(points, (0,2,1))
        points, target = points.cuda(), target.cuda()
        
        classifier = classifier.train()
        #forward pass
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        if classifier.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(batchSize)))
        training_df= training_df.append({"epoch": epoch, "i":i, "num_batch":num_batch, "train_loss":loss.item(), "accuracy":correct.item()/float(batchSize)},ignore_index=True)
        if i % 10 == 0:
            j, data = next(enumerate(validationloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, 'validation', loss.item(), correct.item()/float(batchSize)))
            validation_df = validation_df.append({"epoch": epoch, "i":i, "num_batch":num_batch, "validation_loss":loss.item(), "accuracy":correct.item()/float(batchSize)},ignore_index=True)
    torch.save(classifier.state_dict(), '%s/classifier_model_%d.pth' % ("Trained_Model", epoch))
    training_df.to_csv('training.csv')    
    validation_df.to_csv('validation.csv') 


# In[9]:





# In[ ]:


"""
total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))
"""

