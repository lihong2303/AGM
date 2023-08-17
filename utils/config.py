import os
import argparse
from utils.function_tools import boolean_string

Cramed_config = {
    'fusion':'concat',
    'grad_norm_clip':0,
    'mode':'train',
    'drop_rate':0.1,
}

AVMNIST_Condig = {
    'mode':'train',
}

AVE_Config = {
    'fusion':'concat',
    'grad_norm_clip':0,
    'mode':'train',
    "threshold":0.099,
    "LAMBDA":100,
}


UR_Funny_config = {
    'aligned':True,
    'z_norm':False,
    'flatten':False,
    'max_pad':False,
    'max_pad_num':50,
}


MOSEI_Config = {
    'aligned':True,
    'z_norm':False,
    'flatten':False,
    'max_pad':True,
    'max_pad_num':60,
    "layer":6,
    "hidden_size":512,
    "dropout_r":0.1,
    "multi_head":4,
    "ff_size":1024,
    "word_embed_size":300,
    "lang_seq_len":60,
    "audio_seq_len":60,
    "video_seq_len":60,
    "audio_feat_size":80,
    "video_feat_size":512,
    "task":'sentiment',
    "task_binary":True,
    "mode":'train',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',type=str,default=None)
    parser.add_argument('--checkpoint_path',type=str,default=None)
    parser.add_argument('--breakpoint_path',type=str,default=None)
    parser.add_argument('--device',type=str,choices=['cpu','cuda:0','cuda:1','cuda:2','cuda:3','cuda:4','cuda:5','cuda:6','cuda:7'])
    parser.add_argument('--methods',type=str,choices=['Normal','OGM-GE','AGM',"MSLR","MSES"],help="Methods used for training")
    parser.add_argument('--modality',type=str,default='Multimodal',choices=['Audio','Visual','Text','Multimodal'])
    parser.add_argument('--fusion_type',default="late_fusion",type=str,choices=['early_fusion','late_fusion'],help="The type of fusion function.")
    parser.add_argument('--random_seed',type=int,default=999)
    parser.add_argument('--expt_dir',type=str,default="checkpoint")
    parser.add_argument('--expt_name',type=str,default='test_experiment')
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--EPOCHS',type=int,default=10)
    parser.add_argument('--learning_rate',type=float,default=0.00001)
    parser.add_argument('--dataset',default='AVMNIST',type=str,choices=['MOSEI','CREMAD','URFunny','AVE','AV-MNIST'])
    parser.add_argument('--local_rank',default=-1,type=int,help="node rank for distributed training")
    parser.add_argument('--modulation_starts',default=0,type=int,help="where modulation starts.")
    parser.add_argument('--modulation_ends',default=20,type=int,help="where modulation ends")
    parser.add_argument('--alpha',type=float,default=1.0,help='degree of Gradient Modulation')
    parser.add_argument('--lr_decay_ratio',type=float,default=0.1)
    parser.add_argument('--lr_decay_step',type=int,default=70)
    parser.add_argument('--use_mgpu',type=boolean_string,default=False,help='whether to use multi-gpu or not.')
    parser.add_argument('--gpu_ids', default='0,1,2', type=str, help='GPU ids')
    parser.add_argument('--save_checkpoint',type=boolean_string,default=False,help='whether to save checkpoint or not.')
    parser.add_argument('--optim',default='sgd',type=str,choices=['sgd','adamw','adam'],help='the type of the optimizer')
    parser.add_argument('--lr_scalar',default='lrstep',type=str,choices=['lrstep','cosinestep','cosinestepwarmup'],help='the type of the step learning rate')
    args= parser.parse_args()
    return args


class Config():
    def __init__(self):
        args = parse_args()
        args_dict = vars(args)
        self.add_args(args_dict)
        self.select_model_params()
    def add_args(self,args_dict):
        for arg in args_dict.keys():
            setattr(self,arg,args_dict[arg])
    def select_model_params(self):
        if self.dataset == 'MOSEI':
            self.add_args(MOSEI_Config)
        elif self.dataset == 'CREMAD':
            self.add_args(Cramed_config)
        elif self.dataset == 'URFunny':
            self.add_args(UR_Funny_config)
        elif self.dataset == 'AVE':
            self.add_args(AVE_Config)
        elif self.dataset == 'AV-MNIST':
            self.add_args(AVMNIST_Condig)