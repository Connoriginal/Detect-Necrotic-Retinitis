import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import scipy.io as sio

class SingleTaskDataset(Dataset):
    def __init__(self, mode = 'train', train_type = 'smote', class_type = 1):
        '''
        mode: 'train' ,'valid', 'test'
        train_type: 'smote', 'adasyn', 'smoteenn'
        class_type : 1 (classify 0 & 1), 2 (classify 0 & 2)
        '''
        self.mode = mode
        self.train_type = train_type
        self.type = class_type
        
        if self.mode == 'train':
            if self.train_type == 'smote':
                self.df = pd.read_csv('./data/train_data/smote.csv')
            elif self.train_type == 'adasyn':
                self.df = pd.read_csv('./data/train_data/adasyn.csv')
            elif self.train_type == 'smoteenn':
                self.df = pd.read_csv('./data/train_data/smoteenn.csv')

        elif self.mode == 'valid':
            self.df = pd.read_csv('./data/valid.csv')

        elif self.mode == 'test':
            self.df = pd.read_csv('./data/test.csv')

        df_features = self.df.loc[:, self.df.columns != 'Diagnosis']
        df_labels = self.df['Diagnosis']
        self.feature_names = list(df_features.columns)
        self.features = df_features.values.astype(np.float32)
        self.labels = df_labels.values
        self.labels = np.where(self.labels == class_type, 1, 0)
        
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

