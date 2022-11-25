#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import h5py
import glob
import csv

def load_data(train, validation):
    if ((train== True) and (validation == False)):
        partition = 'train'
    elif((train== True) and (validation == True)):
        partition = 'val'
    else:
        partition = 'test'
    BASE_DIR = os.path.dirname(os.path.abspath('hdf5_data'))
    DATA_DIR = os.path.join(BASE_DIR,'hdf5_data','ply_data_%s*.h5' % partition)
    all_data = []
    all_label = []
    all_point_id = []
    for h5_name in glob.glob(DATA_DIR):
        f = h5py.File(h5_name,'r+')
        data = f['data'][:].astype('float32')
        label= f['label'][:].astype('int64')
        point_id= f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_point_id.append(point_id)
    all_data = torch.from_numpy(np.concatenate(all_data, axis = 0))
    all_label = torch.from_numpy(np.concatenate(all_label, axis= 0))
    all_point_id = torch.from_numpy(np.concatenate(all_point_id, axis= 0))
    return all_data,all_label,all_point_id    

#a,b,c = load_data(train = False, validation = False)
#print(c.size())

class ShapeNetData(Dataset):
    def __init__(self, train = True, validation = False, num_points = 2048, randomize_data = False):
        super(ShapeNetData, self).__init__()
        self.data,self.labels,self.point_id = load_data(train, validation)
        self.num_points = num_points
        self.randomize_data = randomize_data
        if not train:
            self.shapes = self.read_classes_ShapeNet()
            
    def __getitem__(self,idx):
        if self.randomize_data:
            current_points = self.randomize(idx)
        else:
            current_points = self.data[idx]
        #current_points = torch.from_numpy(current_points[:self.num_points, :]).float()
        #label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)
        current_points =(current_points[:self.num_points, :]).float()
        label = (self.labels[idx])
        point_id = (self.point_id[idx])
        return current_points, label,point_id
    
    def __len__(self):
        return self.data.shape[0]
    
    def randomize(self,idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)
        return self.data[idx,pt_idxs].copy()
    def read_classes_ShapeNet(self):
        BASE_DIR = os.path.dirname(os.path.abspath('hdf5_data'))
        DATA_DIR = os.path.join(BASE_DIR,'hdf5_data')
        file = open(os.path.join(DATA_DIR, 'all_object_categories_Copy.txt'), 'r')
        shape_names = file.read()
        shape_names= np.array(shape_names.split('\n')[:-1])
        return shape_names


#D = ShapeNetData(train = True, validation = False, num_points = 2048, randomize_data = False)
#print(D.point_id.size())

