import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    """
    Extend to nn.Transformer.
    """
    def __init__(self,n_features,dim,n_head,n_layers):
        super(Transformer,self).__init__()
        self.embed_dim = dim
        self.conv = nn.Conv1d(n_features,self.embed_dim,kernel_size=1,padding=0,bias=False)
        layer = nn.TransformerEncoderLayer(self.embed_dim,nhead=n_head)
        self.transformer = nn.TransformerEncoder(layer,num_layers=n_layers)


    def forward(self,x):
        """
        Apply transorformer to input tensor.

        """
        if type(x) is list:
            x = x[0]
        x = self.conv(x.permute([0,2,1]))
        x = x.permute([2,0,1])
        x = self.transformer(x)[0]
        return x

class MLP(nn.Module):
    """
    Two layer perception.
    """
    def __init__(self,indim,hiddim,outdim,one_layer=False,dropout=False,dropoutp=0.1,output_each_layer=False):
        super(MLP,self).__init__()
        self.one_layer = one_layer
        if self.one_layer:
            self.fc = nn.Linear(indim,outdim)
        else:
            self.fc = nn.Linear(indim,hiddim)
            self.fc2 = nn.Linear(hiddim,outdim)
            self.dropout_layer = nn.Dropout(dropoutp)
            self.dropout = dropout
            self.output_each_layer = output_each_layer
            self.lklu = nn.LeakyReLU(0.2)

    def forward(self,x):
        """
        Apply MLP to input Tensor.
        """
        if self.one_layer:
            return self.fc(x)
        else:
            output = F.relu(self.fc(x))
            if self.dropout:
                output = self.dropout_layer(output)
            output2 = self.fc2(output)
            if self.dropout:
                output2 = self.dropout_layer(output2)

            if self.output_each_layer:
                return [0,x,output,self.lklu(output2)]
            return output2
        
class MaxOut_MLP(nn.Module):
    def __init__(self,num_outputs,first_hidden=64,num_input_feats = 300,seconed_hidden=None,linear_layer=True):
        super(MaxOut_MLP,self).__init__()
        if seconed_hidden is None:
            seconed_hidden = first_hidden
        self.bn1 = nn.BatchNorm1d(num_input_feats,1e-4)
        self.maxout1 = Maxout(num_input_feats,first_hidden,2)
        self.module1 = nn.Sequential(nn.BatchNorm1d(first_hidden),nn.Dropout(0.3))
        self.maxout2 = Maxout(first_hidden,seconed_hidden,2)
        self.module2 = nn.Sequential(nn.BatchNorm1d(seconed_hidden),nn.Dropout(0.3))
        
        if linear_layer:
            self.fc = nn.Linear(seconed_hidden,num_outputs)
        else:
            self.fc = None
            
    def forward(self,x):
        bn1_out = self.bn1(x)
        max1_out = self.maxout1(bn1_out)
        module1_out = self.module1(max1_out)
        max2_out = self.maxout2(module1_out)
        out = self.module2(max2_out)
        if self.fc is not None:
            out = self.fc(out)
            
        return out

class Maxout(nn.Module):
    def __init__(self,d,m,k):
        super(Maxout,self).__init__()
        self.d_in,self.d_out,self.pool_size = d,m,k
        self.fc = nn.Linear(d,m*k)
        
    def forward(self,inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.fc(inputs)
        m,_ = out.view(*shape).max(dim=max_dim)
        return m