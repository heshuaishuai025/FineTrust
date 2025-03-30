import math
import torch
import torch.nn.functional as F
import random
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.init as init

import numpy as np
from torch.nn import Parameter

def uniform(size, tensor):  
    """
    Uniform weight initialization.权重初始化
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

class ListModule(torch.nn.Module):
    """
    Abstract list layer class. 抽象列表层类
    """
    def __init__(self, *args):
        """
        Model initializing.  模型初始化
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.  获取索引层
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.  迭代
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers. 层数
        """
        return len(self._modules)

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape, device=torch.device("cuda"))
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class Convolution(torch.nn.Module):
    """
    Abstract Signed SAGE convolution class.
    :param in_channels: Number of features.   特征
    :param out_channels: Number of filters.     过滤器
    :param norm_embed: Normalize embedding -- boolean.  嵌入
    :param bias: Add bias or no.  偏置
    """
    def __init__(self,
                 in_channels, 
                 out_channels,
                 num_labels, 
                 aggregation_mean= False,  #平均聚合器
                 norm_embed=False,  #标准嵌入
                 bias=True):  #偏置
        super(Convolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregation_mean = aggregation_mean
        self.norm_embed = norm_embed
        self.num_labels = num_labels
        self.slope_ratio = 0.1
        self.requires_grad=False
        self.speical_spmm = SpecialSpmm()
        
        self.weight_base= Parameter(torch.Tensor(self.in_channels, int(out_channels/2))) 
        self.weight = Parameter(torch.Tensor(int(self.in_channels*2), int(out_channels/2)))   #48*64
        self.trans_weight = Parameter(torch.Tensor(int(self.num_labels/2), int(self.in_channels/2)))
     
        self.a_base = Parameter(torch.Tensor(in_channels*2, 1))
        self.a = Parameter(torch.Tensor(in_channels*4, 1))
        
        nn.init.kaiming_normal_(self.a.data)
        nn.init.kaiming_normal_(self.a_base.data)


        nn.init.kaiming_normal_(self.weight_base.data)
        nn.init.kaiming_normal_(self.weight.data)
        nn.init.kaiming_normal_(self.trans_weight.data)
    
        if bias:
            self.bias = Parameter(torch.Tensor(int(out_channels/2)))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        size = self.weight.size(0)
        uniform(size, self.bias)
        size3 = self.weight_base.size(0)
        # size2 = self.trans_weight.size(0)
        size4 = self.a.size(0)
        size5 = self.a_base.size(0)
        uniform(size, self.weight)
        uniform(size3, self.weight_base)
       

        # uniform(size2, self.trans_weight)
        uniform(size5, self.a_base)
        uniform(size4, self.a)

    def __repr__(self):
        """
        Create formal string representation.
        """
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)

class ConvolutionBase_out(Convolution):     #K考虑  人气信任+参与信任
    """
    Base Signed SAGE class for the first layer of the model.
    """
    def forward(self, x, edge_index, labels):
        """
        Forward propagation pass with features an indices.
        :param x_1: Features for left hand side vertices.
        :param x_2: Features for right hand side vertices.
        :param edge_index: Positive indices.
        :param edge_index_neg: Negative indices.
        :return out: Abstract convolved features.
        """
        row, col = edge_index  # 正边索引(起始节点-中止节点)
        index = edge_index.t()
        edge_h_2 = torch.cat((x[col], x[row]), dim=1)  # Whi||Whj   25623*128
        edges_h = torch.exp(F.leaky_relu(torch.mm(edge_h_2, self.a_base), self.slope_ratio)) # attention    ##relu后存在inf
        edges_h = torch.mul(edges_h, labels)
        # edges_h = torch.exp(F.leaky_relu(torch.einsum("ij,jl->il", [edge_h_2, self.a_base])))
        row_sum = self.speical_spmm(index.t(), edges_h[:, 0], torch.Size((x.shape[0],x.shape[0])), torch.ones(size=(x.shape[0], 1)).to(torch.device("cuda")))
        results = self.speical_spmm(index.t(), edges_h[:, 0], torch.Size((x.shape[0],x.shape[0])), x)
        row_sum.clamp_(1e-6)
        # 乘完特征再去除
        output_emb = results.div(row_sum)
        h = torch.matmul(output_emb, self.weight_base)
        # if self.bias is not None:
        #     h = h + self.bias
        return h
 
    
class ConvolutionBase_in(Convolution):     #K考虑  人气信任+参与信任
    """
    Base Signed SAGE class for the first layer of the model.
    """
    def forward(self, x, edge_index,labels):
        """
        Forward propagation pass with features an indices.
        :param x_1: Features for left hand side vertices.
        :param x_2: Features for right hand side vertices.
        :param edge_index: Positive indices.
        :param edge_index_neg: Negative indices.
        :return out: Abstract convolved features.
        """
        row, col = edge_index   # 正边索引(起始节点-中止节点)
      
        index = edge_index.t()
        edge_h_2 = torch.cat((x[row], x[col]), dim=1)  # Whi||Whj
        edges_h = torch.exp(F.leaky_relu(torch.mm(edge_h_2, self.a_base), self.slope_ratio)) # attention    ##relu后存在inf
        edges_h = torch.mul(edges_h, labels)
        # edges_h = torch.exp(F.leaky_relu(torch.einsum("ij,jl->il", [edge_h_2, self.a_base])))
        row_sum = self.speical_spmm(index.t(), edges_h[:, 0], torch.Size((x.shape[0],x.shape[0])), torch.ones(size=(x.shape[0], 1)).to(torch.device("cuda")))
        results = self.speical_spmm(index.t(), edges_h[:, 0], torch.Size((x.shape[0],x.shape[0])), x)
        row_sum.clamp_(1e-6)
        # 乘完特征再去除
        output_emb = results.div(row_sum)
        h = torch.matmul(output_emb, self.weight_base)
        # 
        
        return h




class ConvolutionDeep_pos_in(Convolution):
    """
    Deep Signed SAGE class for multi-layer models.
    """
    def forward(self, x, x_1, x_2, edge_index_pos, edge_index_neg, labels):
        """
        Forward propagation pass with features an indices.
        :param x_1: Features for left hand side vertices.
        :param x_2: Features for right hand side vertices.
        :param edge_index: Positive indices.
        :param edge_index_neg: Negative indices.
        :return out: Abstract convolved features.
        """
        row_pos, col_pos = edge_index_pos  # 正边索引(起始节点-中止节点)
        row_neg, col_neg = edge_index_neg  # 正边索引(起始节点-中止节点)
        # labels_trans = torch.matmul(labels, self.trans_weight)
       
        index_pos = edge_index_pos.t()
        edge_h_1 = torch.cat((x[row_pos], x[col_pos]), dim=1)  # Whu||Whm   28563*128
        edges_h_a = torch.exp(F.leaky_relu(torch.mm(edge_h_1, self.a), self.slope_ratio)) # attention   
        edges_h_a = torch.mul(edges_h_a, labels)   # trustworthiness
        # edges_h = torch.exp(F.leaky_relu(torch.einsum("ij,jl->il", [edge_h_2, self.a_base])))
        row_sum_1 = self.speical_spmm(index_pos.t(), edges_h_a[:, 0], torch.Size((x.shape[0],x.shape[0])), torch.ones(size=(x.shape[0], 1)).to(torch.device("cuda")))
        results_1 = self.speical_spmm(index_pos.t(), edges_h_a[:, 0], torch.Size((x.shape[0],x.shape[0])), x)
        row_sum_1.clamp_(1e-6)
        # 乘完特征再去除
        h = results_1.div(row_sum_1)
        
        h = torch.matmul(h, self.weight)
        if self.bias is not None:
            h = h + self.bias
        return h


class ConvolutionDeep_neg_in(Convolution):
    """
    Deep Signed SAGE class for multi-layer models.
    """
    def forward(self, x, x_1, x_2, edge_index_pos, edge_index_neg, labels):
        """
        Forward propagation pass with features an indices.
        :param x_1: Features for left hand side vertices.
        :param x_2: Features for right hand side vertices.
        :param edge_index: Positive indices.
        :param edge_index_neg: Negative indices.
        :return out: Abstract convolved features.
        """
        row_pos, col_pos = edge_index_pos  # 正边索引(起始节点-中止节点)
        row_neg, col_neg = edge_index_neg  # 正边索引(起始节点-中止节点)
        # labels_trans = torch.matmul(labels, self.trans_weight)
       
        index_neg = edge_index_neg.t()
        edge_h_1 = torch.cat((x[row_neg], x[col_neg]), dim=1)  # Whu||Whn
        edges_h_a = torch.exp(F.leaky_relu(torch.mm(edge_h_1, self.a), self.slope_ratio)) # attention   
        edges_h_a = torch.mul(edges_h_a, labels)   # trustworthiness
        # edges_h = torch.exp(F.leaky_relu(torch.einsum("ij,jl->il", [edge_h_2, self.a_base])))
        row_sum_1 = self.speical_spmm(index_neg.t(), edges_h_a[:, 0], torch.Size((x.shape[0],x.shape[0])), torch.ones(size=(x.shape[0], 1)).to(torch.device("cuda")))
        results_1 = self.speical_spmm(index_neg.t(), edges_h_a[:, 0], torch.Size((x.shape[0],x.shape[0])), x)
        row_sum_1.clamp_(1e-6)
        # 乘完特征再去除
        h = results_1.div(row_sum_1)
      
        h = torch.matmul(h, self.weight)
        if self.bias is not None:
            h = h + self.bias
        return h

  

class ConvolutionDeep_pos_out(Convolution):
    """
    Deep Signed SAGE class for multi-layer models.
    """
    def forward(self, x, x_1, x_2, edge_index_pos, edge_index_neg, labels):
        """
        Forward propagation pass with features an indices.
        :param x_1: Features for left hand side vertices.
        :param x_2: Features for right hand side vertices.
        :param edge_index: Positive indices.
        :param edge_index_neg: Negative indices.
        :return out: Abstract convolved features.
        """
        row_pos, col_pos = edge_index_pos  # 正边索引(起始节点-中止节点)
        row_neg, col_neg = edge_index_neg  # 正边索引(起始节点-中止节点)
        # labels_trans = torch.matmul(labels, self.trans_weight)
       
        index_pos = edge_index_pos.t()
        edge_h_1 = torch.cat((x[row_pos], x[col_pos]), dim=1)  # Whu||Whp
        edges_h_a = torch.exp(F.leaky_relu(torch.mm(edge_h_1, self.a), self.slope_ratio)) # attention   
        edges_h_a = torch.mul(edges_h_a, labels)   # trustworthiness
        # edges_h = torch.exp(F.leaky_relu(torch.einsum("ij,jl->il", [edge_h_2, self.a_base])))
        row_sum_1 = self.speical_spmm(index_pos.t(), edges_h_a[:, 0], torch.Size((x.shape[0],x.shape[0])), torch.ones(size=(x.shape[0], 1)).to(torch.device("cuda")))
        results_1 = self.speical_spmm(index_pos.t(), edges_h_a[:, 0], torch.Size((x.shape[0],x.shape[0])), x)
        row_sum_1.clamp_(1e-6)
        # 乘完特征再去除
        h = results_1.div(row_sum_1)
        

        h = torch.matmul(h, self.weight)
        if self.bias is not None:
            h = h + self.bias
        return h
    
    
class ConvolutionDeep_neg_out(Convolution):
    """
    Deep Signed SAGE class for multi-layer models.
    """
    def forward(self, x, x_1, x_2, edge_index_pos, edge_index_neg, labels):
        """
        Forward propagation pass with features an indices.
        :param x_1: Features for left hand side vertices.
        :param x_2: Features for right hand side vertices.
        :param edge_index: Positive indices.
        :param edge_index_neg: Negative indices.
        :return out: Abstract convolved features.
        """
        row_pos, col_pos = edge_index_pos  # 正边索引(起始节点-中止节点)
        row_neg, col_neg = edge_index_neg  # 正边索引(起始节点-中止节点)
        # labels_trans = torch.matmul(labels, self.trans_weight)
       
        index_neg = edge_index_neg.t()
        edge_h_1 = torch.cat((x[row_neg], x[col_neg]), dim=1)  # Whu||Whq
        edges_h_a = torch.exp(F.leaky_relu(torch.mm(edge_h_1, self.a), self.slope_ratio)) # attention   
        edges_h_a = torch.mul(edges_h_a, labels)   # trustworthiness
        # edges_h = torch.exp(F.leaky_relu(torch.einsum("ij,jl->il", [edge_h_2, self.a_base])))
        row_sum_1 = self.speical_spmm(index_neg.t(), edges_h_a[:, 0], torch.Size((x.shape[0],x.shape[0])), torch.ones(size=(x.shape[0], 1)).to(torch.device("cuda")))
        results_1 = self.speical_spmm(index_neg.t(), edges_h_a[:, 0], torch.Size((x.shape[0],x.shape[0])), x)
        row_sum_1.clamp_(1e-6)
        # 乘完特征再去除
        h = results_1.div(row_sum_1)
        
        h = torch.matmul(h, self.weight)
        if self.bias is not None:
            h = h + self.bias
        return h