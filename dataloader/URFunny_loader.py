import torch
import pickle
import numpy as np
import torch.nn.functional as F
from typing import *
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence

class URFunny_Dataloader(object):
    def __init__(self,cfgs):
        self.cfgs = cfgs
        self.train_dataset = Humor_Dataset(cfgs,mode='train')
        self.valid_dataset = Humor_Dataset(cfgs,mode='valid')
        self.test_dataset = Humor_Dataset(cfgs,mode='test')
        self.process = eval('_process_2') if cfgs.max_pad else eval("_process_1")

    @property
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.cfgs.batch_size,
                          shuffle=True,
                          num_workers=32,
                          pin_memory=True,
                          collate_fn=self.process)
    @property
    def valid_dataloader(self):
        return DataLoader(dataset=self.valid_dataset,
                          batch_size=self.cfgs.batch_size,
                          shuffle=False,
                          num_workers=32,
                          pin_memory=True,
                          collate_fn=self.process)
    @property
    def dep_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.cfgs.batch_size,
                          shuffle=False,
                          num_workers=32,
                          pin_memory=True,
                          collate_fn=self.process)

def _process_1(inputs:List):
    processed_input = []
    processed_input_lengths = []
    inds = []
    labels = []

    for i in range(len(inputs[0])-2):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(torch.as_tensor([v.size(0) for v in feature]))
        pad_seq  = pad_sequence(feature,batch_first=True)
        processed_input.append(pad_seq)

    for sample in inputs:
        inds.append(sample[-2])

        if sample[-1].shape[1] > 1:
            labels.append(sample[-1].reshape(sample[-1].shape[1],sample[-1].shape[0])[0])
        else:
            labels.append(sample[-1])

    return processed_input,processed_input_lengths,torch.tensor(inds).view(len(inputs),1),torch.tensor(labels).view(len(inputs),1)

def _process_2(inputs:List):
    processed_input = []
    processed_input_length = []
    labels = []

    for i in range(len(inputs[0])-1):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_length.append(torch.as_tensor([v.size(0) for v in feature]))
        processed_input.append(torch.stack(feature))
    
    for sample in inputs:
        if sample[-1].shape[1] > 1:
            labels.append(sample[-1].reshape(sample[-1].shape[1],sample[-1].shape[0])[0])
        else:
            labels.append(sample[-1])
    return processed_input[0],processed_input[1],processed_input[2],torch.tensor(labels).view(len(inputs),1)


def drop_entry(dataset):
    """
    drop entries where there's no text in the data.
    """
    drop = []
    for idx,k in enumerate(dataset['text']):
        if k.sum() == 0:
            drop.append(idx)
    for modality in list(dataset.keys()):
        dataset[modality] = np.delete(dataset[modality],drop,0)
    return dataset

class Humor_Dataset(Dataset):
    def __init__(self,args,mode='train',task=None):
        data_root = args.data_root

        self.aligned = args.aligned
        self.z_norm = args.z_norm
        self.task = task
        self.flatten = args.flatten
        self.max_pad = args.max_pad
        self.max_pad_num = args.max_pad_num

        with open(data_root,'rb') as f:
            all_data = pickle.load(f)

        assert mode in ['train','valid','test']
        self.dataset = drop_entry(all_data[mode])
    
    def __getitem__(self, index):
        vision = torch.tensor(self.dataset['vision'][index])
        audio = torch.tensor(self.dataset['audio'][index])
        text = torch.tensor(self.dataset['text'][index])
        audio[audio==-np.inf]=0.0

        if self.aligned:
            try:
                start = text.nonzero(as_tuple=False)[0][0]
            except:
                print(text,index)
                exit()
            vision = vision[start:].float()
            audio = audio[start:].float()
            text = text[start:].float()
        else:
            vision = vision[vision.nonzero()[0][0]:].float()
            audio = audio[audio.nonzero()[0][0]:].float()
            text = text[text.nonzero()[0][0]:].float()
        # z-normalization data
        if self.z_norm:
            vision = torch.nan_to_num((vision-vision.mean(0)) / (torch.std(vision)))
            audio = torch.nan_to_num((audio-audio.mean(0))/(torch.std(audio)))
            text = torch.nan_to_num((text-text.mean(0)) / (torch.std(text)))
        tmp_label = self.dataset['labels'][index]
        if (self.task==None) or (self.task == 'regression'):
            if self.dataset['labels'][index] < 1:
                tmp_label = [[0]]
            else:
                tmp_label = [[1]]
        
        def _get_class(flag):
            return [flag]

        label = torch.tensor(_get_class(tmp_label)).long() if self.task == "classification" else torch.tensor(tmp_label).float()
        label = label.long()

        if self.flatten:
            return [vision.flatten(),audio.flatten(),text.flatten(),index,label]
        else:
            if self.max_pad:
                tmp = [vision,audio,text,label]
                for i in range(len(tmp)-1):
                    tmp[i] = tmp[i][:self.max_pad_num]
                    tmp[i] = F.pad(tmp[i],(0,0,0,self.max_pad_num-tmp[i].shape[0]))
            else:
                tmp = [vision,audio,text,index,label]
            return tmp

    def __len__(self):
        """
        Get length of dataset.
        """
        return self.dataset['vision'].shape[0]