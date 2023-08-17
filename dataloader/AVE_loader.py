import numpy as np
import torch
import h5py
import pickle
import random
import os
import pdb
from torch.utils.data import Dataset,DataLoader

ave_dataset = ['bell', 'Male', 'Bark', 'aircraft', 'car', 'Female', 'Helicopter',
    'Violin', 'Flute', 'Ukulele', 'Fry food', 'Truck', 'Shofar', 'Motorcycle',
    'guitar', 'Train', 'Clock', 'Banjo', 'Goat', 'Baby', 'Bus',
    'Chainsaw', 'Cat', 'Horse', 'Toilet', 'Rodents', 'Accordion', 'Mandolin', 'background']

class AVE_Fully_Dataset(object):
    """Data preparation for fully supervised setting.
    """
    def __init__(self, cfgs,mode):

        super(AVE_Fully_Dataset,self).__init__()
        self.cfgs = cfgs
        self.data_root = cfgs.data_root
        self.video_dir = os.path.join(self.data_root,'visual_feature.h5')
        self.audio_dir = os.path.join(self.data_root,'audio_feature.h5')
        self.label_dir = os.path.join(self.data_root,'right_labels.h5')
        if mode == "train":
            self.order_dir = os.path.join(self.data_root,'train_order.h5')
        elif mode == "val":
            self.order_dir = os.path.join(self.data_root,'val_order.h5')
        elif mode == "test":
            self.order_dir = os.path.join(self.data_root,'test_order.h5')
        else:
            raise NotImplementedError
        self.batch_size = cfgs.batch_size
        # self.status = status

        with h5py.File(self.audio_dir, 'r') as hf:
            self.audio_features = hf['avadataset'][:]
        with h5py.File(self.label_dir, 'r') as hf:
            self.labels = hf['avadataset'][:]
        with h5py.File(self.video_dir, 'r') as hf:
            self.video_features = hf['avadataset'][:]

        with h5py.File(self.order_dir, 'r') as hf:
            order = hf['order'][:]

        self.lis = order.tolist()
        self.list_copy = self.lis.copy().copy()

        self.video_batch = np.float32(np.zeros([self.batch_size, 10, 7, 7, 512]))
        self.audio_batch = np.float32(np.zeros([self.batch_size, 10, 128]))
        self.pos_audio_batch = np.float32(np.zeros([self.batch_size, 10, 128]))
        self.label_batch = np.float32(np.zeros([self.batch_size, 10, 29]))
        self.segment_label_batch = np.float32(np.zeros([self.batch_size, 10]))
        self.segment_avps_gt_batch = np.float32(np.zeros([self.batch_size, 10]))

    def get_segment_wise_relation(self, batch_labels):
        bs, seg_num, category_num = batch_labels.shape
        all_seg_idx = list(range(seg_num))
        for i in range(bs):
            col_sum = np.sum(batch_labels[i].T, axis=1)
            category_bg_cols = col_sum.nonzero()[0].tolist()
            category_bg_cols.sort()

            category_col_idx = category_bg_cols[0]
            category_col = batch_labels[i, :, category_col_idx]
            same_category_row_idx = category_col.nonzero()[0].tolist()
            if len(same_category_row_idx) != 0:
                self.segment_avps_gt_batch[i, same_category_row_idx] = 1 / (len(same_category_row_idx))

        for i in range(bs):
            row_idx, col_idx = np.where(batch_labels[i] == 1)
            self.segment_label_batch[i, row_idx] = col_idx


    def __len__(self):
        return len(self.lis)


    def get_batch(self, idx, shuffle_samples=False):
        if shuffle_samples:
            random.shuffle(self.list_copy)
        select_ids = self.list_copy[idx * self.batch_size : (idx + 1) * self.batch_size]

        for i in range(self.batch_size):
            id = select_ids[i]
            v_id = id
            self.video_batch[i, :, :, :, :] = self.video_features[v_id, :, :, :, :]
            self.audio_batch[i, :, :] = self.audio_features[id, :, :]
            self.label_batch[i, :, :] = self.labels[id, :, :]

        self.get_segment_wise_relation(self.label_batch)


        return torch.from_numpy(self.audio_batch).float(), \
                torch.from_numpy(self.video_batch).float(), \
                torch.from_numpy(self.label_batch).float(), \
                torch.from_numpy(self.segment_label_batch).long(), \
                torch.from_numpy(self.segment_avps_gt_batch).float()


