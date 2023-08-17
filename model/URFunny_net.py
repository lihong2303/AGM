import torch

from torch import nn,Tensor
from model.utils.module_base import Transformer,MLP,MaxOut_MLP

class Modality_Text(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,total_out,
                pad_visual_out,pad_audio_out,pad_text_out,
                pad_visual_audio_out,pad_visual_text_out,pad_audio_text_out,
                zero_padding_out):
        return (total_out-pad_text_out+pad_visual_audio_out)/3 + (pad_visual_out - pad_audio_text_out+pad_audio_out-pad_visual_text_out)/6

class Modality_Audio(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,total_out,
                pad_visual_out,pad_audio_out,pad_text_out,
                pad_visual_audio_out,pad_visual_text_out,pad_audio_text_out,
                zero_padding_out):
        return (total_out-pad_audio_out+pad_visual_text_out) / 3 + (pad_visual_out - pad_audio_text_out + pad_text_out - pad_visual_audio_out) / 6

class Modality_Visual(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,total_out,
                pad_visual_out,pad_audio_out,pad_text_out,
                pad_visual_audio_out,pad_visual_text_out,pad_audio_text_out,
                zero_padding_out):
        return (total_out-pad_visual_out+pad_audio_text_out)/3 + (pad_audio_out-pad_visual_text_out + pad_text_out - pad_visual_audio_out)/6

class Modality_out(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x

class GradMod(nn.Module):
    def __init__(self,cfgs):
        super().__init__()
        self.extract_mm_feature = False
        self.mode = 'train'
        if cfgs.fusion_type == "late_fusion":
            self.net = Later_Fusion_Model(cfgs)
        elif cfgs.fusion_type == "early_fusion":
            self.net = Early_Fusion_Model(cfgs)
        self.m_t = Modality_Text()
        self.m_a = Modality_Audio()
        self.m_v = Modality_Visual()
        self.m_v_o = Modality_out()
        self.m_a_o = Modality_out()
        self.m_t_o = Modality_out()

        self.scale_a = 1.0
        self.scale_v = 1.0
        self.scale_t = 1.0

        self.m_a_o.register_full_backward_hook(self.hooka)
        self.m_v_o.register_full_backward_hook(self.hookv)
        self.m_t_o.register_full_backward_hook(self.hookt)

    def hooka(self,m,ginp,gout):
        gnew = ginp[0].clone()
        return gnew * self.scale_a,
    
    def hookv(self,m,ginp,gout):
        gnew = ginp[0].clone()
        return gnew*self.scale_v,
    
    def hookt(self,m,ginp,gout):
        gnew = ginp[0].clone()
        return gnew*self.scale_t,

    def update_scale(self,coeff_a,coeff_v,coeff_t):
        self.scale_a = coeff_a
        self.scale_v = coeff_v
        self.scale_t = coeff_t

    def forward(self,vision,audio,text,feature_length):
        """
        Args:
            vision:vision modality feature;
            audio:audio modality feature;
            text:text modality feature;
            feature_length:each feature length.

        Returns:
            m_a,m_v,m_t:audio,visual and text modality output,respectively(no zero padding).
            c_a,c_v,c_t:audio,visual and text modality marginal contribution,respectively(containinf zero padding).
            out:model out.
        """
        self.net.mode = "feature"
        total_out,encoded_feature = self.net(vision,audio,text,feature_length,pad_audio = False,pad_visual=False,pad_text=False)
        self.net.mode = "classify"
        self.net.eval()
        pad_audio_out = self.net(vision,audio,text,feature_length,pad_audio = True,pad_visual=False,pad_text=False)
        pad_visual_out = self.net(vision,audio,text,feature_length,pad_audio = False,pad_visual=True,pad_text=False)
        pad_text_out = self.net(vision,audio,text,feature_length,pad_audio = False,pad_visual=False,pad_text=True)
        pad_visual_audio_out = self.net(vision,audio,text,feature_length,pad_audio = True,pad_visual=True,pad_text=False)
        pad_visual_text_out = self.net(vision,audio,text,feature_length,pad_audio = False,pad_visual=True,pad_text=True)
        pad_audio_text_out = self.net(vision,audio,text,feature_length,pad_audio = True,pad_visual=False,pad_text=True)
        zero_padding_out = self.net(vision,audio,text,feature_length,pad_audio = True,pad_visual=True,pad_text=True)
        if self.mode == 'train':
            self.net.train()
        m_a_out = self.m_a_o(self.m_a(total_out,
                pad_visual_out,pad_audio_out,pad_text_out,
                pad_visual_audio_out,pad_visual_text_out,pad_audio_text_out,
                zero_padding_out))
        m_v_out = self.m_v_o(self.m_v(total_out,
                pad_visual_out,pad_audio_out,pad_text_out,
                pad_visual_audio_out,pad_visual_text_out,pad_audio_text_out,
                zero_padding_out))
        m_t_out = self.m_t_o(self.m_t(total_out,
                pad_visual_out,pad_audio_out,pad_text_out,
                pad_visual_audio_out,pad_visual_text_out,pad_audio_text_out,
                zero_padding_out))

        # individual marginal contribution (contain zero padding)
        m_a_mc = m_a_out - zero_padding_out / 3
        m_v_mc = m_v_out - zero_padding_out / 3
        m_t_mc = m_t_out - zero_padding_out /3
        if self.extract_mm_feature is True:
            return m_a_mc,m_v_mc,m_t_mc,m_a_out,m_v_out,m_t_out,m_a_out + m_v_out + m_t_out,encoded_feature
        else:
            return m_a_mc,m_v_mc,m_t_mc,m_a_out,m_v_out,m_t_out,m_a_out + m_v_out + m_t_out


class Later_Fusion_Model(nn.Module):
    """Transformer Later Fusion Model.

    Args:
        nn (_type_): _description_
    """
    def __init__(self,cfgs):
        super(Later_Fusion_Model,self).__init__()
        self.cfgs = cfgs
        self.audio_encoder = Transformer(81,768,4,8)
        self.visual_encoder = Transformer(371,768,4,8)
        self.text_encoder = Transformer(300,768,4,8)
        self.mode = "classify"
        self.head = MLP(2304,1024,2,one_layer=True)
        
        
    def forward(self,vision,audio,text,feature_length,pad_audio=False,pad_visual=False,pad_text=False):
        if pad_audio:
            audio = torch.zeros_like(audio,dtype=audio.dtype,device=audio.device)

        if pad_visual:
            vision = torch.zeros_like(vision,dtype=vision.dtype,device=vision.device)

        if pad_text:
            text = torch.zeros_like(text,dtype=text.dtype,device=text.device)
        audio_encoded_out = self.audio_encoder([audio,feature_length[1]])
        vision_encoded_out = self.visual_encoder([vision,feature_length[0]])
        text_encoded_out = self.text_encoder([text,feature_length[2]])
        
        encoded_out = torch.cat([audio_encoded_out,vision_encoded_out,text_encoded_out],dim=1)
        
        out = self.head(encoded_out)
        if self.mode == "feature":
            return out,encoded_out
        return out
    
class Later_Fusion_Model_Sum(nn.Module):
    """Transformer Later Fusion Model.

    Args:
        nn (_type_): _description_
    """
    def __init__(self,cfgs):
        super(Later_Fusion_Model_Sum,self).__init__()
        self.cfgs = cfgs
        self.extract_layer_conductance = False
        self.extract_mm_feature = False
        self.audio_encoder = Transformer(81,768,4,8)
        self.visual_encoder = Transformer(371,768,4,8)
        self.text_encoder = Transformer(300,768,4,8)
        
        self.audio_cls = nn.Linear(768,2)
        self.visual_cls = nn.Linear(768,2)
        self.text_cls = nn.Linear(768,2)
        
    def forward(self,vision,audio,text,feature_length):
        audio_encoded_out = self.audio_encoder([audio,feature_length[1]])
        vision_encoded_out = self.visual_encoder([vision,feature_length[0]])
        text_encoded_out = self.text_encoder([text,feature_length[2]])
        
        encoded_out = torch.cat([audio_encoded_out,vision_encoded_out,text_encoded_out],dim=1)
        
        out_a = self.audio_cls(audio_encoded_out)
        out_v = self.visual_cls(vision_encoded_out)
        out_t = self.text_cls(text_encoded_out)

        out = out_a + out_v + out_t
        if self.extract_layer_conductance is True:
            return out
        
        if self.extract_mm_feature is True:
            return out_a,out_v,out_t,out,encoded_out
        return out_a,out_v,out_t,out

class Audio_Transformer(nn.Module):
    """
    """
    def __init__(self,cfgs):
        super(Audio_Transformer,self).__init__()
        self.cfgs = cfgs

        self.audio_encoder = Transformer(81,768,4,8)
        self.head = MLP(768,128,2,one_layer=True)
    def forward(self,audio,feature_length):
        audio_encoded_out = self.audio_encoder([audio,feature_length[1]])
        out = self.head(audio_encoded_out)
        return out

class Visual_Transformer(nn.Module):
    def __init__(self,cfgs):
        super(Visual_Transformer,self).__init__()
        self.cfgs = cfgs
        self.visual_encoder = Transformer(371,768,4,8)
        self.head = MLP(768,128,2,one_layer=True)
    def forward(self,visual,feature_length):
        visual_encoded_out = self.visual_encoder([visual,feature_length[0]])
        out = self.head(visual_encoded_out)
        return out

class Text_Transformer(nn.Module):
    def __init__(self,cfgs):
        super(Text_Transformer,self).__init__()
        self.cfgs = cfgs
        self.text_encoder = Transformer(300,768,4,8)
        self.head = MLP(768,128,2,one_layer=True)
    def forward(self,text,feature_length):
        text_encoded_out = self.text_encoder([text,feature_length[2]])
        out = self.head(text_encoded_out)
        return out


class Early_Fusion_Model(nn.Module):
    """Transformer Early Fusion Model.

    Args:
        nn (_type_): _description_
    """
    def __init__(self,cfgs):
        super(Early_Fusion_Model,self).__init__()
        self.mode = "classify"
        self.cfgs = cfgs
        self.audio_encoder = Transformer(81,768,4,8)
        self.visual_encoder = Transformer(371,768,4,8)
        self.text_encoder = Transformer(300,768,4,8)
        self.mm_encoder = MaxOut_MLP(2,2048,2304,1024,linear_layer=False)
        self.head = MLP(1024,128,2,one_layer=True)

    def forward(self,vision,audio,text,feature_length,pad_audio=False,pad_visual=False,pad_text=False):
        if pad_audio:
            audio = torch.zeros_like(audio,dtype=audio.dtype,device=audio.device)

        if pad_visual:
            vision = torch.zeros_like(vision,dtype=vision.dtype,device=vision.device)

        if pad_text:
            text = torch.zeros_like(text,dtype=text.dtype,device=text.device)
        
        audio_encoded_out = self.audio_encoder([audio,feature_length[1]])
        vision_encoded_out = self.visual_encoder([vision,feature_length[0]])
        text_encoded_out = self.text_encoder([text,feature_length[2]])
        mm_out = torch.cat([audio_encoded_out,vision_encoded_out,text_encoded_out],dim=1)
        encoded_out = self.mm_encoder(mm_out)
        out = self.head(encoded_out)
        if self.mode == "feature":
            return out,encoded_out
        return out