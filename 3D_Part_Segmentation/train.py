#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import print_function
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataloader import ShapeNetData
from segmentation_model import STN3d,STNkd,PointNetfeature,PointNetSegmenter,feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd


# In[4]:


dataset = ShapeNetData(train = True, validation = False, num_points = 2048, randomize_data = False)
test_dataset = ShapeNetData(train = False, validation = False, num_points = 2048, randomize_data = False)
validation_dataset = ShapeNetData(train = True, validation = True, num_points = 2048, randomize_data = False)

dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=0)
testdataloader = torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=True,num_workers=0)
validationloader = torch.utils.data.DataLoader(validation_dataset,batch_size=32,shuffle=True,num_workers=0)

#print(len(dataset), len(validation_dataset), len(test_dataset))


# In[5]:


#num_class = 50
model =  PointNetSegmenter(k=50, feature_transform=True) #k= num_classes
FILE = "model.pth"
torch.save(model.state_dict(),FILE)


# In[7]:


part_segmenter = PointNetSegmenter(k=50, feature_transform=True)   #k= num_classes
part_segmenter.load_state_dict(torch.load(FILE))
part_segmenter.cuda()


# In[8]:


optimizer = optim.Adam(part_segmenter.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


# In[10]:


try:
    os.makedirs("Trained_Seg_Model")
except OSError:
    pass


# In[11]:


training_df= pd.DataFrame(columns=["epoch", "i", "num_batch", "train_loss", "accuracy"] )
validation_df= pd.DataFrame(columns=["epoch", "i", "num_batch", "validation_loss", "accuracy"] )
    


# In[19]:


batchSize = 32
num_batch = len(dataset) /batchSize
num_epoch = 120
for epoch in range(num_epoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points,class_label,target = data
        points = points.transpose(2, 1)
        points,class_label,target = points.cuda(),class_label.cuda(),target.cuda()
        part_segmenter = part_segmenter.train()
        #forwardpass
        pred, trans, trans_feat = part_segmenter(points)
        pred = pred.view(-1, 50)
        target = target.view(-1, 1)[:, 0]
        #print(pred.size(), target.size())
        loss = F.nll_loss(pred, target)
        if part_segmenter.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        #Backwardpass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(batchSize * 2048)))
        training_df= training_df.append({"epoch": epoch, "i":i, "num_batch":num_batch, "train_loss":loss.item(), "accuracy":correct.item()/float(batchSize * 2048)},ignore_index=True)
        if i % 10 == 0:
            j, data = next(enumerate(validationloader, 0))
            points,class_label, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            part_segmenter = part_segmenter.eval()
            pred, _, _ = part_segmenter(points)
            pred = pred.view(-1, 50)
            target = target.view(-1, 1)[:, 0]
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, 'validation', loss.item(), correct.item()/float(batchSize * 2048)))
            validation_df = validation_df.append({"epoch": epoch, "i":i, "num_batch":num_batch, "validation_loss":loss.item(), "accuracy":correct.item()/float(batchSize * 2048)},ignore_index=True)
    torch.save(part_segmenter.state_dict(), 'Trained_Seg_Model/seg_model_%d.pth' % (epoch))
    training_df.to_csv('training.csv')    
    validation_df.to_csv('validation.csv') 


# In[48]:


"""
seg_parts = {0:[0,1,2,3],1:[4,5],2:[6,7],3:[8,9,10,11], 4: [12,13,14,15], '5': [16,17,18],'6': [19,20,21],'7':[22,23],'8':[24,25,26,27],'9':[28,29],
             '10':[30,31,32,33,34,35],'11':[36,37],'12':[38,39,40],'13':[41,42,43],'14':[44,45,46],'15':[47,48,49]}
"""


# In[49]:


#print(seg_parts.keys())


# In[45]:


"""
## benchmark mIOU
shape_ious = []
for i,data in tqdm(enumerate(testdataloader)):
    points,class_label,target = data
    points = points.transpose(2, 1)
    points,target = points.cuda(),target.cuda()
    part_segmenter = part_segmenter.eval()
    pred, _ , _ = part_segmenter(points)
    pred_choice = pred.data.max(2)[1]
    pred_np = pred_choice.cpu().data.numpy() 
    target_np = target.cpu().data.numpy()
    #print(target_np.shape[1])
    #print(class_label)

    for shape_idx in range(target_np.shape[0]):
        print(shape_idx)
        cat_label = int(class_label[shape_idx])
        print(cat_label)
        parts = seg_parts[cat_label]
        print(parts)
     
        print('iteration',shape_idx,cat_label)
        parts = seg_parts[cat_label]
        print(parts)
 
        parts = range(50)#np.unique(target_np[shape_idx])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
        
print("mIOU for shapes{}".format(np.mean(shape_ious)))
"""


# In[ ]:




