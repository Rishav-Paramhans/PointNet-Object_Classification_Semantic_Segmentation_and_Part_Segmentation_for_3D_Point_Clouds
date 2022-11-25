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
    BASE_DIR = os.path.dirname(os.path.abspath('indoor3d_sem_seg_hdf5_data'))
    DATA_DIR = os.path.join(BASE_DIR,'indoor3d_sem_seg_hdf5_data','ply_data_all_%s*.h5' % partition)
    all_data = []
    all_label = []
    for h5_name in glob.glob(DATA_DIR):
        f = h5py.File(h5_name,'r+')
        data = f['data'][:].astype('float32')
        label= f['label'][:].astype('int64')
        all_data.append(data)
        all_label.append(label)
    all_data = torch.from_numpy(np.concatenate(all_data ,axis = 0))
    all_label = torch.from_numpy(np.concatenate(all_label, axis = 0))
    return all_data,all_label


class S3DIS(Dataset):
    def __init__(self, train = True, validation = False, randomize_data = False):
        super(S3DIS, self).__init__()
        self.data,self.labels= load_data(train, validation)
        self.randomize_data = randomize_data
        if not train:
            self.shapes = self.read_object_S3DIS()
            
    def __getitem__(self,idx):
        if self.randomize_data:
            current_points = self.randomize(idx)
        else:
            current_points = self.data[idx]
        #current_points = torch.from_numpy(current_points[:self.num_points, :]).float()
        #label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)
        current_points =(current_points[:4096, :]).float()
        label = (self.labels[idx])
        return current_points, label
    
    def __len__(self):
        return self.data.shape[0]
    
    def randomize(self,idx):
        pt_idxs = np.arange(0, 4096)
        np.random.shuffle(pt_idxs)
        return self.data[idx,pt_idxs].copy()
    def read_object_S3DIS(self):
        BASE_DIR = os.path.dirname(os.path.abspath('indoor3d_sem_seg_hdf5_data'))
        DATA_DIR = os.path.join(BASE_DIR,'indoor3d_sem_seg_hdf5_data')
        file = open(os.path.join(DATA_DIR, 'object_cateorgory.txt'), 'r')
        object_names = file.read()
        object_names= np.array(object_names.split('\n')[:-1])
        return object_names
