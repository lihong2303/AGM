import torch.optim as optim
from model.CREMAD_net import AClassify,VClassify,AVClassifier_Sum,GradMod
from dataloader.CREMAD_loader import Cramed_DataLoader

class Cramed_Task(object):
    def __init__(self,cfgs):
        super(Cramed_Task,self).__init__()
        self.cfgs = cfgs
        self.train_dataloader,self.test_dataloader,self.dep_dataloader = self.load_dataloader()
        self.model = self.build_model()
        self.optimizer,self.scheduler = self.build_optimizer()

    def load_dataloader(self):
        loader = Cramed_DataLoader(self.cfgs)
        train_loader = loader.train_dataloader
        test_loader = loader.test_dataloader
        dep_loader = loader.dep_dataloader
        return train_loader,test_loader,dep_loader

    def build_model(self):
        if self.cfgs.fusion_type == "early_fusion":
            net = GradMod(self.cfgs)
        else:
            if self.cfgs.modality == 'Multimodal':
                if self.cfgs.methods == 'AGM':
                    net = GradMod(self.cfgs)
                else:
                    net = AVClassifier_Sum(self.cfgs)
            elif self.cfgs.modality == 'Audio':
                net = AClassify(self.cfgs)
            elif self.cfgs.modality == 'Visual':
                net = VClassify(self.cfgs)
        return net

    def build_optimizer(self):
        optimizer = optim.SGD(self.model.parameters(),lr = self.cfgs.learning_rate,momentum=0.9,weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer,self.cfgs.lr_decay_step,self.cfgs.lr_decay_ratio)
        return optimizer,scheduler