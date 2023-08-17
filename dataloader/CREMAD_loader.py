import csv
import os
import librosa
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

class Cramed_DataLoader(DataLoader):
    def __init__(self,cfgs):
        self.cfgs = cfgs
        if cfgs.dataset == 'CREMAD':
            self.train_dataset = CreamdDataset(cfgs,mode='train')
            self.test_dataset = CreamdDataset(cfgs,mode='test')
            self.dep_dataset = CreamdDataset(cfgs,mode="dep")
        else:
            raise NotImplementedError('Incorrect dataset name {}!'.format(cfgs.dataset))
        
    @property
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                            batch_size = self.cfgs.batch_size,
                            shuffle=True,
                            num_workers=32,
                            pin_memory=True)
    @property
    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size = self.cfgs.batch_size,
                          shuffle=False,
                          num_workers=32,
                          pin_memory=True)
        
    @property
    def dep_dataloader(self):
        return DataLoader(dataset=self.dep_dataset,
                          batch_size=self.cfgs.batch_size,
                          shuffle=False,
                          num_workers=32,
                          pin_memory=True)

class CreamdDataset(Dataset):

    def __init__(self, args, mode='train'):
        super(CreamdDataset,self).__init__()
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.item = []
        self.mode = mode
        self.data_root = args.data_root
        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.visual_feature_path = args.data_root
        self.audio_feature_path = os.path.join(self.data_root,'Audio')

        self.train_csv = os.path.join(self.data_root, args.dataset + '/train.csv')
        self.test_csv = os.path.join(self.data_root, args.dataset + '/test.csv')

        self.train_image = []
        self.train_audio = []
        self.train_label = []
        self.train_item = []
        
        self.test_image = []
        self.test_audio = []
        self.test_label = []
        self.test_item = []
        with open(self.train_csv, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
                visual_path = os.path.join(self.visual_feature_path, 'Image-01-FPS', item[0])
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.train_item.append(item[0])
                    self.train_image.append(visual_path)
                    self.train_audio.append(audio_path)
                    self.train_label.append(class_dict[item[1]])
                else:
                    continue
        with open(self.test_csv, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
                visual_path = os.path.join(self.visual_feature_path, 'Image-01-FPS', item[0])
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.test_item.append(item[0])
                    self.test_image.append(visual_path)
                    self.test_audio.append(audio_path)
                    self.test_label.append(class_dict[item[1]])
                else:
                    continue
        if mode == "train":
            self.item = self.train_item[:int(len(self.train_item)*0.9)]
            self.image = self.train_image[:int(len(self.train_image)*0.9)]
            self.audio = self.train_audio[:int(len(self.train_audio)*0.9)]
            self.label = self.train_label[:int(len(self.train_label)*0.9)]
        elif mode == "test":
            self.item = self.test_item[:int(len(self.test_item)*0.9)]
            self.image = self.test_image[:int(len(self.test_image)*0.9)]
            self.audio = self.test_audio[:int(len(self.test_audio)*0.9)]
            self.label = self.test_label[:int(len(self.test_label)*0.9)]
        elif mode == "dep":
            self.item = self.train_item[int(len(self.train_item)*0.9):]
            for k in self.test_item[int(len(self.test_item)*0.9):]:
                self.item.append(k)
            self.image = self.train_image[int(len(self.train_image)*0.9):]
            for k in self.test_image[int(len(self.test_image)* 0.9):]:
                self.image.append(k)
            
            self.audio = self.train_audio[int(len(self.train_image)*0.9):]
            for k in self.test_audio[int(len(self.test_audio)*0.9):]:
                self.audio.append(k)
            
            self.label = self.train_label[int(len(self.train_label)*0.9):]
            for k in self.test_label[int(len(self.test_label)*0.9):]:
                self.label.append(k)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        samples, rate = librosa.load(self.audio[index], sr=22050)
        resamples = np.tile(samples, 3)[:22050*3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        image_samples = os.listdir(self.image[index])
        image_samples = sorted(image_samples)
        images = torch.zeros((1,3,224,224))
        np.random.seed(999)
        select_index = np.random.choice(len(image_samples),1,replace=False)
        img = Image.open(os.path.join(self.image[index],image_samples[int(select_index)])).convert('RGB')
        img = transform(img)
        images[0] = img
        images = images.permute(1,0,2,3)

        label = self.label[index]
        item = self.item[index]
        return item,spectrogram, images, label