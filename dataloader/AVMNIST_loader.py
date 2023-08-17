import torch
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader

class AVMNIST_Loader(object):
    def __init__(self,cfgs):
        super(AVMNIST_Loader,self).__init__()
        self.cfgs = cfgs
        
        self.train_dataset = AVMNIST(cfgs,mode='train')
        self.valid_dataset = AVMNIST(cfgs,mode='test')
        self.dep_dataset = AVMNIST(cfgs,mode='dep')
    @property
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.cfgs.batch_size,
                          shuffle=True,
                          num_workers=64,
                          pin_memory=True)
    
    @property
    def test_dataloader(self):
        return DataLoader(dataset=self.valid_dataset,
                          batch_size=self.cfgs.batch_size,
                          shuffle=False,
                          num_workers=64,
                          pin_memory=True)
    @property
    def dep_dataloader(self):
        return DataLoader(dataset=self.dep_dataset,
                          batch_size=self.cfgs.batch_size,
                          shuffle=False,
                          num_workers=64,
                          pin_memory=True)

class AVMNIST(Dataset):
    def __init__(self,cfgs,mode = 'train') -> None:
        super(AVMNIST,self).__init__()
        self.cfgs = cfgs
        self.data_root = cfgs.data_root
        image_data_path = os.path.join(self.data_root,'image')
        audio_data_path = os.path.join(self.data_root,'audio')
        
        if mode == 'train':
            self.train_image = np.load(os.path.join(image_data_path,'train_data.npy'))
            self.train_audio = np.load(os.path.join(audio_data_path,'train_data.npy'))
            self.train_label = np.load(os.path.join(self.data_root,'train_labels.npy'))
            self.image = self.train_image[:int(self.train_image.shape[0]*0.9)]
            self.audio = self.train_audio[:int(self.train_audio.shape[0]*0.9)]
            self.label = self.train_label[:int(self.train_label.shape[0]*0.9)]
        elif mode == 'test':
            self.test_image = np.load(os.path.join(image_data_path,'test_data.npy'))
            self.test_audio = np.load(os.path.join(audio_data_path,'test_data.npy'))
            self.test_label = np.load(os.path.join(self.data_root,'test_labels.npy'))
            self.image = self.test_image[:int(self.test_image.shape[0]*0.9)]
            self.audio = self.test_audio[:int(self.test_audio.shape[0]*0.9)]
            self.label = self.test_label[:int(self.test_label.shape[0]*0.9)]
        elif mode == 'dep':
            self.train_image = np.load(os.path.join(image_data_path,'train_data.npy'))
            self.train_audio = np.load(os.path.join(audio_data_path,'train_data.npy'))
            self.train_label = np.load(os.path.join(self.data_root,'train_labels.npy'))
            
            self.test_image = np.load(os.path.join(image_data_path,'test_data.npy'))
            self.test_audio = np.load(os.path.join(audio_data_path,'test_data.npy'))
            self.test_label = np.load(os.path.join(self.data_root,'test_labels.npy'))
        
            self.image = np.concatenate([self.train_image[int(self.train_image.shape[0]*0.9):],self.test_image[int(self.test_image.shape[0]*0.9):]])
            self.audio = np.concatenate([self.train_audio[int(self.train_audio.shape[0]*0.9):],self.test_audio[int(self.test_audio.shape[0]*0.9):]])
            self.label = np.concatenate([self.train_label[int(self.train_label.shape[0]*0.9):],self.test_label[int(self.test_label.shape[0]*0.9):]])
        self.length = len(self.image)
        
    def __getitem__(self, index):
        image = self.image[index]
        audio = self.audio[index]
        label = self.label[index]
        
        # Normalize image
        image = image / 255.0
        audio = audio / 255.0
        
        image = image.reshape(28,28)
        image = np.expand_dims(image,0)
        audio = np.expand_dims(audio,0)
        
        image = torch.from_numpy(image)
        audio = torch.from_numpy(audio)
        return audio,image,label
    
    def __len__(self):
        return self.length