import torch
import torch.nn as nn
from model.utils.module_base import MLP,MaxOut_MLP
from model.utils.resnet import resnet18
import torch.nn.functional as F


class Modality_Visual(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,total_out,pad_visual_out,pad_audio_out):
        return 0.5*(total_out-pad_visual_out+pad_audio_out)

class Modality_Audio(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,total_out,pad_visual_out,pad_audio_out):
        return 0.5*(total_out-pad_audio_out+pad_visual_out)

class Modality_out(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x

class GradMod(nn.Module):
    def __init__(self,cfgs):
        super().__init__()
        self.mode = cfgs.mode
        self.extract_mm_feature = False
        if cfgs.fusion_type == "early_fusion":
            self.net = Early_Fusion(cfgs)
        elif cfgs.fusion_type == "late_fusion":
            self.net = Later_Fusion_Resnet(cfgs)
        self.m_v = Modality_Visual()
        self.m_a = Modality_Audio()
        self.m_v_o = Modality_out()
        self.m_a_o = Modality_out()

        self.scale_a = 1.0
        self.scale_v = 1.0

        self.m_a_o.register_full_backward_hook(self.hooka)
        self.m_v_o.register_full_backward_hook(self.hookv)

    def hooka(self,m,ginp,gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_a,
    
    def hookv(self,m,ginp,gout):
        gnew = ginp[0].clone()
        return gnew*self.scale_v,

    def update_scale(self,coeff_a,coeff_v):
        self.scale_a = coeff_a
        self.scale_v = coeff_v

    def forward(self,audio,visual):
        self.net.mode = "feature"
        total_out,feature = self.net(audio,visual,pad_audio = False,pad_visual=False)
        self.net.mode = "classify"
        self.net.eval()
        pad_visual_out = self.net(audio,visual,pad_audio = False,pad_visual=True)
        pad_audio_out = self.net(audio,visual,pad_audio = True,pad_visual=False)
        if self.mode=="train":
            self.net.train()
        m_a = self.m_a_o(self.m_a(total_out,pad_visual_out,pad_audio_out))
        m_v = self.m_v_o(self.m_v(total_out,pad_visual_out,pad_audio_out))
        if self.extract_mm_feature is True:
            return m_a,m_v,m_a + m_v,feature
        elif self.extract_mm_feature is False:
            return m_a,m_v,m_a + m_v  
        

class Early_Fusion(nn.Module):
    def __init__(self,cfgs):
        super(Early_Fusion,self).__init__()
        self.cfgs = cfgs
        self.mode = 'classify'
        self.audio_encoder = resnet18(modality='audio')
        self.visual_encoder = resnet18(modality='image')
        self.mm_encoder = MaxOut_MLP(num_outputs=10,first_hidden=2048,num_input_feats=1024,seconed_hidden=1024,linear_layer=False)
        self.head = MLP(1024,128,10,one_layer=True)
    def forward(self,audio,image,pad_audio=False,pad_visual=False):
        if pad_audio:
            audio = torch.zeros_like(audio,dtype=audio.dtype,device=audio.device)
        if pad_visual:
            image = torch.zeros_like(image,dtype=image.dtype,device=image.device)
        audio_encoded_out = self.audio_encoder(audio)
        image_encoded_out = self.visual_encoder(image)
        audio_out = F.adaptive_avg_pool2d(audio_encoded_out,1)
        image_out = F.adaptive_avg_pool2d(image_encoded_out,1)
        
        audio_out = torch.flatten(audio_out,start_dim=1)
        image_out = torch.flatten(image_out,start_dim=1)
        
        encoded_out = torch.cat((audio_out,image_out),dim=1)
        encoded_out = self.mm_encoder(encoded_out)
        out = self.head(encoded_out)
        if self.mode == 'feature':
            return out,encoded_out
        return out
    
class Later_Fusion_Resnet(nn.Module):
    def __init__(self,cfgs):
        super(Later_Fusion_Resnet,self).__init__()
        self.cfgs = cfgs
        
        self.audio_encoder = resnet18(modality='audio')
        self.visual_encoder = resnet18(modality='image')
        self.mode = 'classify'
        self.head = MLP(1024,100,10,one_layer=True)
        
    def forward(self,audio,image,pad_audio=False,pad_visual=False):
        if pad_audio:
            audio = torch.zeros_like(audio,dtype=audio.dtype,device=audio.device)
        if pad_visual:
            image = torch.zeros_like(image,dtype=image.dtype,device=image.device)
        audio_encoded_out = self.audio_encoder(audio)
        image_encoded_out = self.visual_encoder(image)
        audio_out = F.adaptive_avg_pool2d(audio_encoded_out,1)
        image_out = F.adaptive_avg_pool2d(image_encoded_out,1)
        
        audio_out = torch.flatten(audio_out,start_dim=1)
        image_out = torch.flatten(image_out,start_dim=1)
        
        encoded_out = torch.cat((audio_out,image_out),dim=1)
        out = self.head(encoded_out)
        if self.mode == 'feature':
            return out,encoded_out
        return out

class Later_Fusion_Resnet_Sum(nn.Module):
    def __init__(self,cfgs):
        super(Later_Fusion_Resnet_Sum,self).__init__()
        self.cfgs = cfgs
        self.extract_layer_conductance = False
        self.extract_mm_feature = False
        self.audio_encoder = resnet18(modality='audio')
        self.visual_encoder = resnet18(modality='image')
        self.audio_cls = nn.Linear(512,10)
        self.visual_cls = nn.Linear(512,10)
        
    def forward(self,audio,image):
        audio_encoded_out = self.audio_encoder(audio)
        image_encoded_out = self.visual_encoder(image)
        audio_out = F.adaptive_avg_pool2d(audio_encoded_out,1)
        image_out = F.adaptive_avg_pool2d(image_encoded_out,1)
        
        audio_out = torch.flatten(audio_out,start_dim=1)
        image_out = torch.flatten(image_out,start_dim=1)
        
        encoded_out = torch.cat((audio_out,image_out),dim=1)
        out_a = self.audio_cls(audio_out)
        out_v = self.visual_cls(image_out)

        out = out_a + out_v
        if self.extract_layer_conductance is True:
            return out
        if self.extract_mm_feature is True:
            return out_a,out_v,out,encoded_out
        return out_a,out_v,out
    
class Audio_Classify_ResNet(nn.Module):
    def __init__(self,cfgs):
        super(Audio_Classify_ResNet,self).__init__()
        self.cfgs = cfgs
        
        self.audio_encoder = resnet18(modality='audio')
        self.head = MLP(512,256,10,one_layer=True)
        
    def forward(self,audio):
        audio_encoded_out = self.audio_encoder(audio)
        audio_out = F.adaptive_avg_pool2d(audio_encoded_out,1)
        
        audio_out = torch.flatten(audio_out,start_dim=1)
        out = self.head(audio_out)
        return out
    

class Visual_Classify_ResNet(nn.Module):
    def __init__(self,cfgs):
        super(Visual_Classify_ResNet,self).__init__()
        
        self.cfgs = cfgs
        self.visual_encoder = resnet18(modality='image')
        self.head = MLP(512,256,10,one_layer=True)
        
    def forward(self,visual):
        B = visual.size(0)
        visual_encoded_out = self.visual_encoder(visual)
        (T,C,H,W) = visual_encoded_out.size()
        visual_encoded_out = visual_encoded_out.view(B,-1,C,H,W)
        visual_encoded_out = visual_encoded_out.permute(0,2,1,3,4)
        visual_out = F.adaptive_avg_pool2d(visual_encoded_out,1)
        visual_out = torch.flatten(visual_out,1)
        
        out = self.head(visual_out)
        
        return out