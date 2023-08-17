import torch.optim as optim
from dataloader.AVMNIST_loader import AVMNIST_Loader
from model.AVMNIST_net import GradMod,Audio_Classify_ResNet,Visual_Classify_ResNet,Later_Fusion_Resnet_Sum

class AVMNIST_Task(object):
    def __init__(self,cfgs):
        super(AVMNIST_Task,self).__init__()
        self.cfgs = cfgs
        self.train_dataloader,self.test_dataloader,self.dep_dataloader = self.load_dataloader()
        self.model = self.load_model()
        self.optimizer,self.scheduler = self.build_optimizer()
        
    def load_dataloader(self):
        loader = AVMNIST_Loader(self.cfgs)
        train_dataloader = loader.train_dataloader
        test_dataloader = loader.test_dataloader
        dep_dataloader = loader.dep_dataloader
        return train_dataloader,test_dataloader,dep_dataloader
    
    def load_model(self):
        if self.cfgs.fusion_type == 'late_fusion':
            if self.cfgs.modality == 'Audio':
                net = Audio_Classify_ResNet(self.cfgs)
            elif self.cfgs.modality == 'Visual':
                net = Visual_Classify_ResNet(self.cfgs)
            elif self.cfgs.modality == 'Multimodal':
                if self.cfgs.methods =='AGM':
                    net = GradMod(self.cfgs)
                else:
                    net = Later_Fusion_Resnet_Sum(self.cfgs)
            else:
                raise NotImplementedError
        elif self.cfgs.fusion_type == 'early_fusion':
            net = GradMod(self.cfgs)
        return net
    
    def build_optimizer(self):
        optimizer = optim.SGD(self.model.parameters(),lr = self.cfgs.learning_rate,momentum=0.9,weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer,self.cfgs.lr_decay_step,self.cfgs.lr_decay_ratio)
        return optimizer,scheduler