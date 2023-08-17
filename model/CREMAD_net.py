import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.module_base import MaxOut_MLP,MLP
from model.utils.resnet import resnet18

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
        if cfgs.fusion_type == 'late_fusion':
            self.net = AVClassifier(cfgs)
        elif cfgs.fusion_type == 'early_fusion':
            self.net = AV_Early_Classifier(cfgs)
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
        total_out,encoded_feature = self.net(audio,visual,pad_audio = False,pad_visual=False)
        self.net.mode = "classify"
        self.net.eval()
        pad_visual_out = self.net(audio,visual,pad_audio = False,pad_visual=True)
        pad_audio_out = self.net(audio,visual,pad_audio = True,pad_visual=False)
        zero_padding_out = self.net(audio,visual,pad_audio = True,pad_visual=True)
        if self.mode=="train":
            self.net.train()
        m_a = self.m_a_o(self.m_a(total_out,pad_visual_out,pad_audio_out))
        m_v = self.m_v_o(self.m_v(total_out,pad_visual_out,pad_audio_out))

        if self.extract_mm_feature is True:
            return total_out,pad_visual_out,pad_audio_out,zero_padding_out,m_a + m_v,encoded_feature
        return total_out,pad_visual_out,pad_audio_out,zero_padding_out,m_a + m_v
    

class AVClassifier(nn.Module):
    """ Using ResNet as audio & image encoder for late-fusion model.

    Args:
        nn (_type_): _description_
    """
    def __init__(self,args):
        super(AVClassifier,self).__init__()
        self.mode = "classify"
        self.args = args
        if args.dataset == 'CREMAD':
            n_classes = 6
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if args.fusion == "concat":
            input_dim = 1024
            self.fusion_module = ConcatFusion(input_dim=input_dim,output_dim=n_classes)

        else:
            raise NotImplementedError(f'Incorrect fusion method:{args.fusion}')

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

    def forward(self,audio,visual,pad_audio=False,pad_visual=False):
        if pad_audio:
            audio = torch.zeros_like(audio,device=audio.device)
        
        if pad_visual:
            visual = torch.zeros_like(visual,device=visual.device)

        a = self.audio_net(audio)
        v = self.visual_net(visual)
        (T,C,H,W) = v.size()
        B = a.size()[0]
        v = v.view(B,-1,C,H,W)
        v = v.permute(0,2,1,3,4)

        a = F.adaptive_avg_pool2d(a,1)
        v = F.adaptive_avg_pool3d(v,1)
        a = torch.flatten(a,1)
        v = torch.flatten(v,1)

        out,encoded_feature = self.fusion_module(a,v)
        if self.mode == "feature":
            return out,encoded_feature
        return out
class AVClassifier_Sum(nn.Module):
    """ Using ResNet as audio & image encoder for late-fusion model.

    Args:
        nn (_type_): _description_
    """
    def __init__(self,args):
        super(AVClassifier_Sum,self).__init__()
        self.extract_mm_feature = False
        self.args = args

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')
        self.audio_cls = nn.Linear(512,6)
        self.visual_cls = nn.Linear(512,6)

    def forward(self,audio,visual,pad_audio=False,pad_visual=False):
        if pad_audio:
            audio = torch.zeros_like(audio,device=audio.device)
        
        if pad_visual:
            visual = torch.zeros_like(visual,device=visual.device)

        a = self.audio_net(audio)
        v = self.visual_net(visual)
        (T,C,H,W) = v.size()
        B = a.size()[0]
        v = v.view(B,-1,C,H,W)
        v = v.permute(0,2,1,3,4)

        a = F.adaptive_avg_pool2d(a,1)
        v = F.adaptive_avg_pool3d(v,1)
        a = torch.flatten(a,1)
        v = torch.flatten(v,1)
        feature = torch.cat((a,v),dim=1)
        out_a = self.audio_cls(a)
        out_v = self.visual_cls(v)
        out = out_a + out_v
        if self.extract_mm_feature is True:
            return out_a,out_v,out,feature
        return out_a,out_v,out
class AV_Early_Classifier(nn.Module):
    """Using Resnet as muliti-modal encoder for early-fusion model.

    Args:
        nn (_type_): _description_
    """
    def __init__(self,args):
        super(AV_Early_Classifier,self).__init__()
        self.mode = "classify"
        self.args = args
        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')
        self.mm_encoder = MaxOut_MLP(6,2048,1024,1024,linear_layer=False)
        self.head = MLP(1024,128,6,one_layer=True)
    def forward(self,audio,visual,pad_audio=False,pad_visual=False):
        if pad_audio:
            audio = torch.zeros_like(audio,device=audio.device)
        
        if pad_visual:
            visual = torch.zeros_like(visual,device=visual.device)

        a = self.audio_net(audio)
        v = self.visual_net(visual)
        (T,C,H,W) = v.size()
        B = a.size()[0]
        v = v.view(B,-1,C,H,W)
        v = v.permute(0,2,1,3,4)

        a = F.adaptive_avg_pool2d(a,1)
        v = F.adaptive_avg_pool3d(v,1)
        a = torch.flatten(a,1)
        v = torch.flatten(v,1)
        
        encoded_out = torch.cat([a,v],dim=1)
        feature = self.mm_encoder(encoded_out)
        out = self.head(feature)
        if self.mode == "feature":
            return out,feature
        return out
    
class AClassify(nn.Module):
    def __init__(self,args):
        super(AClassify,self).__init__()
        self.args = args
        self.mode = "classify"
        if args.dataset == 'CREMAD':
            n_classes = 6
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))
        input_dim = 512
        self.cls = ClassifyLayer(input_dim,n_classes)
        self.audio_net = resnet18(modality='audio')
    def forward(self,audio):
        a = self.audio_net(audio)
        a = F.adaptive_avg_pool2d(a,1)
        a = torch.flatten(a,1)
        out = self.cls(a)
        if self.mode == "feature":
            return a
        return out
    

class VClassify(nn.Module):
    def __init__(self,args):
        super(VClassify,self).__init__()
        self.args = args
        self.mode = "classify"
        if args.dataset == 'CREMAD':
            n_classes = 6
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        input_dim = 512
        self.cls = ClassifyLayer(input_dim=input_dim,output_dim=n_classes)
        self.visual_net = resnet18(modality='visual')
    
    def forward(self,v):
        B = v.size(0)
        v = self.visual_net(v)
        (T,C,H,W) = v.size()
        
        v = v.view(B,-1,C,H,W)
        v = v.permute(0,2,1,3,4)
        v = F.adaptive_avg_pool3d(v,1)

        v = torch.flatten(v,1)

        out = self.cls(v)
        if self.mode == "feature":
            return v
        return out

class ConcatFusion(nn.Module):
    def __init__(self,input_dim=1024,output_dim=100):
        super(ConcatFusion,self).__init__()
        self.fc_out = nn.Linear(input_dim,output_dim)

    def forward(self,x,y):
        encoded_output = torch.cat((x,y),dim=1)
        output = self.fc_out(encoded_output)

        return output,encoded_output
    



class  ClassifyLayer(nn.Module):
    def __init__(self,input_dim=512,output_dim=100):
        super(ClassifyLayer,self).__init__()
        self.fc = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        output = self.fc(x)
        return output