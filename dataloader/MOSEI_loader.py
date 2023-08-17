import os
import torch
import pickle
import numpy as np

from torch.utils.data import Dataset,DataLoader
from dataloader.tokenizer import *

class Mosei_Loader(object):
    def __init__(self,args):
        self.args = args
        self.train_dataset = Mosei_Dataset('train',self.args,None)
        self.valid_dataset = Mosei_Dataset('valid',self.args,self.train_dataset.token_to_ix)
        self.test_dataset = Mosei_Dataset('test',self.args,self.train_dataset.token_to_ix)

    @property
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.args.batch_size,
                          shuffle=True,
                          num_workers=32,
                          pin_memory=True)
    @property
    def valid_dataloader(self):
        return DataLoader(dataset=self.valid_dataset,
                          batch_size=self.args.batch_size,
                          shuffle=False,
                          num_workers=32,
                          pin_memory=True)

    @property
    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.args.batch_size,
                          shuffle=False,
                          num_workers=32,
                          pin_memory=True)
    

class Mosei_Dataset(Dataset):
    def __init__(self,name,args,token_to_ix=None) -> None:
        super(Mosei_Dataset,self).__init__()
        assert name in ["train","valid","test","private"]
        self.args = args
        self.private_set = name == "private"
        self.dataroot = args.data_root

        word_file = os.path.join(self.dataroot,name+"_sentences.p")
        audio_file = os.path.join(self.dataroot,name+"_mels.p")
        video_file = os.path.join(self.dataroot,name+"_mels.p")
        # video_file = os.path.join(self.dataroot,name + "r21d.p")

        y_s_file = os.path.join(self.dataroot,name + "_sentiment.p")
        y_e_file = os.path.join(self.dataroot,name+"_emotion.p")

        self.set = eval(name.upper() + "_SET")

        self.key_to_word = pickle.load(open(word_file,"rb"))
        self.key_to_audio = pickle.load(open(audio_file,"rb"))
        self.key_to_video = pickle.load(open(video_file,"rb"))

        # If private test,labels dont exist.
        if not self.private_set:
            if args.task == "emotion":
                self.key_to_label = pickle.load(open(y_e_file,"rb"))
            if args.task == "sentiment":
                self.key_to_label = pickle.load(open(y_s_file,"rb"))
            
            for key in self.set:
                if not (key in self.key_to_word and
                        key in self.key_to_audio and
                        key in self.key_to_video and
                        key in self.key_to_label):
                    self.set.remove(key)
        for key in self.set:
            Y = self.key_to_label[key]

        # Creating embeddings and word indexes
        self.key_to_sentence = tokenize(self.key_to_word)
        if token_to_ix is not None:
            self.token_to_ix = token_to_ix
        else:
            self.token_to_ix,self.pretrained_emb = create_dict(self.key_to_sentence,self.dataroot)
        
        self.vocab_size = len(self.token_to_ix)

        self.l_max_len = args.lang_seq_len
        self.a_max_len = args.audio_seq_len
        self.v_max_len = args.video_seq_len

    def __getitem__(self, index):
        key = self.set[index]
        L = sent_to_ix(self.key_to_sentence[key],self.token_to_ix,max_token=self.l_max_len)
        A = pad_feature(self.key_to_audio[key],self.a_max_len)
        V = pad_feature(self.key_to_video[key],self.v_max_len)
        
        y = np.array([])
        if not self.private_set:
            Y = self.key_to_label[key]
            # print(Y)
            if self.args.task == "sentiment" and self.args.task_binary:
                c = cmumosei_2(Y)
                y = np.array(c)
            elif self.args.task == "sentiment" and not self.args.task_binary:
                c = cmumosei_7(Y)
                y = np.array(c)
            elif self.args.task == "emotion":
                Y[Y>0] = 1
                y = Y

        return key,torch.from_numpy(L),torch.from_numpy(A),torch.from_numpy(V).float(),torch.from_numpy(y)

    def __len__(self):
        return len(self.set)