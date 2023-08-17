from torch.optim import SGD,AdamW,lr_scheduler,Adam
from model.MOSEI_net import GradMod
from dataloader.MOSEI_loader import Mosei_Loader

class Mosei_Task(object):
    def __init__(self,cfgs):
        super(Mosei_Task,self).__init__()
        self.cfgs = cfgs
        self.train_dataloader,self.valid_dataloader,self.test_dataloader = self.load_dataloader()
        self.model = self.build_model()
        self.optimizer,self.scheduler = self.build_optimizer()

    def load_dataloader(self):
        loader = Mosei_Loader(self.cfgs)
        train_dataloader = loader.train_dataloader
        valid_dataloader = loader.valid_dataloader
        test_dataloader = loader.test_dataloader
        return train_dataloader,valid_dataloader,test_dataloader

    def build_model(self):
        model = GradMod(self.cfgs,self.train_dataloader.dataset.vocab_size,self.train_dataloader.dataset.pretrained_emb)
        return model
        
    def build_optimizer(self):
        if self.cfgs.optim == 'sgd':
            optimizer = SGD(self.model.parameters(),lr=self.cfgs.learning_rate,momentum=0.9,weight_decay=1e-4)
        elif self.cfgs.optim == 'adamw':
            optimizer = AdamW(self.model.parameters(),lr=self.cfgs.learning_rate,weight_decay=1e-4)
        elif self.cfgs.optim == "adam":
            optimizer = Adam(self.model.parameters(),lr=self.cfgs.learning_rate)

        if self.cfgs.lr_scalar == 'lrstep':
            scheduler = lr_scheduler.StepLR(optimizer,self.cfgs.lr_decay_step,self.cfgs.lr_decay_ratio)
        elif self.cfgs.lr_scalar == 'cosinestep':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,eta_min=1e-6,last_epoch=-1)
        elif self.cfgs.lr_scalar == 'cosinestepwarmup':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,eta_min=1e-6,last_epoch=-1)
        return optimizer,scheduler
