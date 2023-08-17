
from torch.optim import SGD,AdamW,Adam,lr_scheduler
from dataloader.AVE_loader import AVE_Fully_Dataset
from model.AVE_net import GradMod


class AVE_task(object):
    
    def __init__(self,cfgs) -> None:
        super(AVE_task,self).__init__()
        self.cfgs = cfgs
        self.train_dataloader,self.valid_dataloader,self.test_dataloader = self.build_loader()
        self.model = self.build_model()
        self.optimizer,self.scheduler = self.build_optimizer()

    def build_loader(self):
        train_dataloader = AVE_Fully_Dataset(self.cfgs,mode="train")
        valid_dataloader = AVE_Fully_Dataset(self.cfgs,mode="val")
        test_dataloader = AVE_Fully_Dataset(self.cfgs,mode="test")
        return train_dataloader,valid_dataloader,test_dataloader
    
    def build_model(self):
        model = GradMod(self.cfgs)
        return model

    def build_optimizer(self):
        if self.cfgs.optim == 'sgd':
            optimizer = SGD(self.model.parameters(),lr=self.cfgs.learning_rate,momentum=0.9,weight_decay=1e-4)
        elif self.cfgs.optim == 'adamw':
            optimizer = AdamW(self.model.parameters(),lr=self.cfgs.learning_rate,weight_decay=1e-4)
        elif self.cfgs.optim == 'adam':
            optimizer = Adam(self.model.parameters(),lr=self.cfgs.learning_rate)

        if self.cfgs.lr_scalar == 'lrstep':
            scheduler = lr_scheduler.StepLR(optimizer,self.cfgs.lr_decay_step,self.cfgs.lr_decay_ratio)
        elif self.cfgs.lr_scalar == 'cosinestep':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,eta_min=1e-6,last_epoch=-1)
        elif self.cfgs.lr_scalar == 'cosinestepwarmup':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,eta_min=1e-6,last_epoch=-1)
        return optimizer,scheduler