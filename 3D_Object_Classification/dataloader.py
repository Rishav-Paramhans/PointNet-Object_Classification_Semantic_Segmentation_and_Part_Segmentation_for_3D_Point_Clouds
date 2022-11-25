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



def load_data(train, validation, use_normals, split_pt= 7872):
    if train:
        partition = 'train'
    else:
        partition = 'test'
    BASE_DIR = os.path.dirname(os.path.abspath('modelnet40_ply_hdf5_2048'))
    DATA_DIR = os.path.join(BASE_DIR,'modelnet40_ply_hdf5_2048','ply_data_%s*.h5' % partition)
    all_data = []
    all_label = []
    for h5_name in glob.glob(DATA_DIR):
        f = h5py.File(h5_name,'r+')
        if use_normals:
            if validation:
                data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1).astype('float32')
        else:
            data = f['data'][:].astype('float32')
        label= f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = torch.from_numpy(np.concatenate(all_data, axis = 0))
    all_label = torch.from_numpy(np.concatenate(all_label, axis= 0))
    if ((train== True) and (validation == True)):
        all_data = all_data[split_pt:][:][:]
        all_label = all_label[split_pt:][:]
    elif ((train== True) and (validation == False)):
        all_data = all_data[0:split_pt][:][:]
        all_label = all_label[0:split_pt][:]
    else:
        pass
    return all_data, all_label     



class ModelNet40Data(Dataset):
    def __init__(self, train = True, validation = False, num_points = 2048, randomize_data = False, use_normals = False):
        super(ModelNet40Data, self).__init__()
        self.data,self.labels = load_data(train, validation,use_normals)
        self.num_points = num_points
        self.randomize_data = randomize_data
        if not train:
            self.shapes = self.read_classes_ModelNet40()
            
    def __getitem__(self,idx):
        if self.randomize_data:
            current_points = self.randomize(idx)
        else:
            current_points = self.data[idx]
        #current_points = torch.from_numpy(current_points[:self.num_points, :]).float()
        #label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)
        current_points =(current_points[:self.num_points, :]).float()
        label = (self.labels[idx])
        return current_points, label
    
    def __len__(self):
        return self.data.shape[0]
    
    def randomize(self,idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)
        return self.data[idx,pt_idxs].copy()
    def read_classes_ModelNet40(self):
        BASE_DIR = os.path.dirname(os.path.abspath('modelnet40_ply_hdf5_2048'))
        DATA_DIR = os.path.join(BASE_DIR,'modelnet40_ply_hdf5_2048')
        file = open(os.path.join(DATA_DIR, 'shape_names.txt'), 'r')
        shape_names = file.read()
        shape_names= np.array(shape_names.split('\n')[:-1])
        return shape_names
    

class ClassificationData(ModelNet40Data):
    def __init__(self, data_class = ModelNet40Data()):
        super(ClassificationData, self).__init__()
    
    def __len__(self):
        return len(self.data_class)
    
    def set_class(self, data_class):
        self.data_class = data_class
    
    def get_shape(self, label):
        try:
            return self.data_class.get_shape(label)
        except:
            return -1
    def __getitem__(self, index):
        return self.data_class[index]

