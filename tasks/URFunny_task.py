
from torch.optim import SGD,AdamW,lr_scheduler
from dataloader.URFunny_loader import URFunny_Dataloader
from model.URFunny_net import GradMod,Audio_Transformer,Visual_Transformer,Text_Transformer,Later_Fusion_Model_Sum

class Funny_Task(object):
    def __init__(self,cfgs):
        super(Funny_Task,self).__init__()
        self.cfgs = cfgs
        self.train_dataloader,self.valid_dataloader,self.dep_dataloader = self.load_dataloader()
        self.model = self.build_model()
        self.optimizer,self.scheduler = self.build_optimizer()

    def load_dataloader(self):
        loader = URFunny_Dataloader(self.cfgs)
        train_dataloader = loader.train_dataloader
        valid_dataloader = loader.valid_dataloader
        dep_dataloader = loader.dep_dataloader
        return train_dataloader,valid_dataloader,dep_dataloader

    def build_model(self):
        if self.cfgs.fusion_type == "late_fusion":
            if self.cfgs.modality == "Audio":
                model = Audio_Transformer(self.cfgs)
            elif self.cfgs.modality == "Visual":
                model = Visual_Transformer(self.cfgs)
            elif self.cfgs.modality == "Text":
                model = Text_Transformer(self.cfgs)
            else:
                if self.cfgs.methods == 'AGM':
                    model = GradMod(self.cfgs)
                else:
                    model = Later_Fusion_Model_Sum(self.cfgs)
        elif self.cfgs.fusion_type == "early_fusion":
            model = GradMod(self.cfgs)
        return model

    def build_optimizer(self):
        if self.cfgs.optim == 'sgd':
            optimizer = SGD(self.model.parameters(),lr=self.cfgs.learning_rate,momentum=0.9,weight_decay=1e-4)
        elif self.cfgs.optim == 'adamw':
            optimizer = AdamW(self.model.parameters(),lr=self.cfgs.learning_rate,weight_decay=1e-4)

        if self.cfgs.lr_scalar == 'lrstep':
            scheduler = lr_scheduler.StepLR(optimizer,self.cfgs.lr_decay_step,self.cfgs.lr_decay_ratio)
        elif self.cfgs.lr_scalar == 'cosinestep':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,eta_min=1e-6,last_epoch=-1)
        elif self.cfgs.lr_scalar == 'cosinestepwarmup':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,eta_min=1e-6,last_epoch=-1)
        return optimizer,scheduler
