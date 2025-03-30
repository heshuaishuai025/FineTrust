import json
import time
import torch
import random
import torch.nn as nn
import numpy as np
# from sklearn.linear_model import Lasso
import pandas as pd
from tqdm import trange
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F
from utils import calculate_auc, setup_features, structured_negative_sampling
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from convolution import ConvolutionBase_in, ConvolutionBase_out, ConvolutionDeep_pos_in, ConvolutionDeep_pos_out,  ConvolutionDeep_neg_in, ConvolutionDeep_neg_out, ListModule  ############
# import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from numpy.random import normal

class GraphConvolutionalNetwork(torch.nn.Module):
    """
    Graph Convolutional Network Class.
    """
    def __init__(self, device, args, X, num_labels):
        super(GraphConvolutionalNetwork, self).__init__()
        """
        GCN Initialization.
        :param device: Device for calculations.
        :param args: Arguments object.  目标参数
        :param X: Node features.  节点特征X
        :param num_labels: Number of labels   标签数量
        """
        self.args = args
        torch.manual_seed(self.args.seed)
        self.device = device
        self.X = X
        self.dropout = self.args.dropout
        self.num_labels = num_labels
        self.setup_layers()

    def setup_layers(self):
        """
        Adding Base Layers, Deep GraphSAGE layers and Regression Parameters if the model is not a single layer model.  回归系数
        """
        self.nodes = range(self.X.shape[0])   # X矩阵的行数   网络中节点的数量    
        self.neurons = self.args.layers    #神经元数量  
        self.layers = len(self.neurons)  # 迭代层数
        # Base SAGE class for the first layer of the model.
        self.positive_in_aggregators, self.negative_out_aggregators,self.positive_out_aggregators, self.negative_in_aggregators = [], [],[], []
        self.base_in_aggregator = ConvolutionBase_in(self.X.shape[1], self.neurons[0],self.num_labels).to(self.device)
        self.base_out_aggregator = ConvolutionBase_out(self.X.shape[1], self.neurons[0],self.num_labels).to(self.device)
        for i in range(1,self.layers):
         
            self.positive_in_aggregators.append(ConvolutionDeep_pos_in(self.neurons[i-1], self.neurons[i], self.num_labels).to(self.device))
            self.negative_out_aggregators.append(ConvolutionDeep_neg_out(self.neurons[i-1], self.neurons[i], self.num_labels).to(self.device))
            self.positive_out_aggregators.append(ConvolutionDeep_pos_out(self.neurons[i-1], self.neurons[i], self.num_labels).to(self.device))
            self.negative_in_aggregators.append(ConvolutionDeep_neg_in(self.neurons[i-1], self.neurons[i], self.num_labels).to(self.device))
        
        self.positive_in_aggregators = ListModule(*self.positive_in_aggregators)
        self.negative_in_aggregators = ListModule(*self.negative_in_aggregators)
        self.positive_out_aggregators = ListModule(*self.positive_out_aggregators)
        self.negative_out_aggregators = ListModule(*self.negative_out_aggregators)
        
        self.regression_weights = Parameter(torch.Tensor(4*self.neurons[-1], self.num_labels))
        self.regression_weight = Parameter(torch.Tensor(2*self.neurons[-1], 2*self.neurons[-1]))  
        self.regression_weights_out = Parameter(torch.Tensor(2*self.neurons[-1], 1))   
    
        init.xavier_normal_(self.regression_weights) 
        init.xavier_normal_(self.regression_weight) 
        init.xavier_normal_(self.regression_weights_out) 
        
    def calculate_positive_embedding_loss(self, z, positive_edges):
        """
        Calculating the loss on the positive edge embedding distances
        :param z: Hidden vertex representation.
        :param positive_edges: Positive training edges.
        :return : Loss value on positive edge embedding.
        """
        i, j, k = structured_negative_sampling(positive_edges,z.shape[0])
        self.positive_z_i = z[i]
        self.positive_z_j = z[j]
        self.positive_z_k = z[k]

        out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def calculate_negative_embedding_loss(self, z, negative_edges):
        """
        Calculating the loss on the negative edge embedding distances
        :param z: Hidden vertex representation.
        :param negative_edges: Negative training edges.
        :return : Loss value on negative edge embedding.
        """
        i, j, k = structured_negative_sampling(negative_edges,z.shape[0])
        self.negative_z_i = z[i]
        self.negative_z_j = z[j]
        self.negative_z_k = z[k]

        out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()
    
    def calculate_sign_embedding_loss(self, z, train_edges, y_train):  
        """
        Calculating the embedding losses, regression loss and weight regularization loss.
        :param z: Node embedding.
        :param train_edges
        :param target: Target vector.
        :return loss: Value of loss.
        """
        l = torch.nn.Sigmoid()

        status = torch.mm(z, self.regression_weights_out) 
        status = l(status)
        difference = status[train_edges[0, :]] - status[train_edges[1, :]]
        difference = difference.squeeze()
        y_train = torch.where(y_train > 0, 1, -1)

        q = torch.where(y_train == -1, torch.clamp(difference, min=0.5), torch.clamp(difference, max=-0.5))

        tmp = q - difference
        loss_term = torch.sum(tmp.square()) / train_edges.size(1)

        return loss_term

    def calculate_regression_loss(self, z,sorted_train_edges, target):
        """
        Calculating the regression loss for all pairs of nodes.
        :param z: Hidden vertex representations.
        :param target: Target vector.
        :return loss_term: Regression loss.
        :return predictions_soft: Predictions for each vertex pair.
        """
        
        features = torch.tensor([]).to(self.device)
        nolink = torch.tensor([]).to(self.device)
    
        for label in range(self.num_labels):
            edge = sorted_train_edges[label]
            start_node, end_node = z[edge[:,0],:],z[edge[:,1],:]
            node_node = torch.cat((start_node, end_node),1)
            features = torch.cat((features, node_node))

        predictions = torch.mm(features, self.regression_weights)
        predictions_soft = F.log_softmax(predictions, dim=1)
        loss_term = F.nll_loss(predictions_soft, target)    #交叉熵损失
        return loss_term
    
    
    
    def calculate_loss_function(self, z, positive_edges, negative_edges, train_edges, y_train,sorted_train_edges, target):
        """
        Calculating the embedding losses, regression loss, and weight regularization loss.
        :param z: Node embedding.   节点嵌入
        :param train_edges  训练边
        :param target: Target vector. 目标向量
        :return loss: Value of loss.
        """ 

        embedding_loss = self.calculate_sign_embedding_loss(z, train_edges, y_train)
        regression_loss = self.calculate_regression_loss(z,sorted_train_edges, target)

        loss_term = self.args.lama*(regression_loss) + self.args.lamb*embedding_loss
        return loss_term

    def forward(self, positive_edges, negative_edges, train_edges, y_train, positive_labels, negative_labels,sorted_train_edges,target):
    # def forward(self, train_edges, master_edge, apprentice_edge, y, y_train):
        """
        Model forward propagation pass. Can fit deep and single layer SGCN models.  前向传播
        :param edges: edges
        :param y: Target vectors. 目标节点
        :return loss: Loss value.
        :return self.z: Hidden vertex representations.  隐藏的向量表示特征Z
        """
        self.h, self.h_pos_in, self.h_pos_out, self.h_neg_in, self.h_neg_out =[], [], [],[],[]
        self.X = F.dropout(self.X, self.dropout, training=self.training)
        
        self.h_pos_in.append(torch.tanh(self.base_in_aggregator(self.X, positive_edges, positive_labels)).to(self.device))
        self.h_pos_out.append(torch.tanh(self.base_out_aggregator(self.X, positive_edges, positive_labels)).to(self.device))
        self.h_neg_in.append(torch.tanh(self.base_in_aggregator(self.X, negative_edges, negative_labels)).to(self.device))
        self.h_neg_out.append(torch.tanh(self.base_out_aggregator(self.X, negative_edges, negative_labels)).to(self.device))
        
        self.h.append(torch.tanh(torch.cat((self.h_pos_in[-1], self.h_pos_out[-1],self.h_neg_in[-1], self.h_neg_out[-1]), 1)).to(self.device))

        for i in range(1,self.layers):
           
            self.h_pos_in.append(torch.tanh(self.positive_in_aggregators[i-1](self.h[i-1], self.h_pos_in[i-1], self.h_neg_in[i-1], positive_edges, negative_edges,positive_labels)).to(self.device))
            self.h_neg_in.append(torch.tanh(self.negative_in_aggregators[i-1](self.h[i-1], self.h_pos_in[i-1], self.h_neg_in[i-1], positive_edges, negative_edges,negative_labels)).to(self.device))
            self.h_pos_out.append(torch.tanh(self.positive_out_aggregators[i-1](self.h[i-1], self.h_pos_out[i-1], self.h_neg_out[i-1], positive_edges, negative_edges,positive_labels)).to(self.device))
            self.h_neg_out.append(torch.tanh(self.negative_out_aggregators[i-1](self.h[i-1], self.h_pos_out[i-1], self.h_neg_out[i-1], positive_edges, negative_edges,negative_labels)).to(self.device))
       
            self.h.append(torch.tanh(torch.matmul(torch.cat((self.h_pos_in[-1], self.h_pos_out[-1],self.h_neg_in[-1], self.h_neg_out[-1]), 1), self.regression_weight)).to(self.device))  #5881*256   256*64
            self.h[-1] = F.dropout(self.h[-1], self.dropout, training=self.training)
            
        self.z = torch.tanh(self.h[-1])
        loss = self.calculate_loss_function(self.z, positive_edges, negative_edges, train_edges, y_train, sorted_train_edges,target)
        return loss, self.z

    
class GCNTrainer(object):
    """
    Object to train and score the SGCN, log the model behaviour and save the output.
    """
    def __init__(self, args, edges):
        """
        Constructing the trainer instance and setting up logs.
        :param args: Arguments object.
        :param edges: Edge data structure
        """
        self.args = args
        self.edges = edges 
        self.device = torch.device("cuda")
        self.global_start_time = time.time()
        self.setup_logs()

    def setup_logs(self):
        """
        Creating a log dictionary for recording performance.
        """
        self.logs = {}
        self.logs["parameters"] =  vars(self.args)
        self.logs["performance"] = [["Epoch", "AUC","F1_micro","F1_macro","F1_weighted","F1","MAE", "RMSE"]]
        self.logs["training_time"] = [["Epoch","Seconds"]]

    def setup_dataset(self):
        """
        Creating train and test split.
        """           
        if self.args.stratified_split:
                stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=self.args.test_size)
                for train_index, test_index in stratSplit.split(self.edges["positive_edges"], self.edges["positive_labels"]):
                    self.positive_edges, self.test_positive_edges = self.edges["positive_edges"][train_index], self.edges["positive_edges"][test_index]
                    self.positive_labels, self.test_positive_labels = self.edges["positive_labels"][train_index], self.edges["positive_labels"][test_index]

                for train_index, test_index in stratSplit.split(self.edges["negative_edges"], self.edges["negative_labels"]):
                    self.negative_edges, self.test_negative_edges = self.edges["negative_edges"][train_index], self.edges["negative_edges"][test_index]
                    self.negative_labels, self.test_negative_labels = self.edges["negative_labels"][train_index], self.edges["negative_labels"][test_index]    
        else:
                self.positive_edges, self.test_positive_edges, self.positive_labels, self.test_positive_labels= train_test_split(self.edges["positive_edges"], self.edges["positive_labels"],
                                                                                        test_size=self.args.test_size)
                self.negative_edges, self.test_negative_edges, self.negative_labels, self.test_negative_labels = train_test_split(self.edges["negative_edges"], self.edges["negative_labels"],
                                                                                        test_size=self.args.test_size)


        self.pos = np.concatenate((self.positive_edges, self.positive_labels), axis = 1)
        self.neg = np.concatenate((self.negative_edges, self.negative_labels), axis = 1)
        self.pos_neg = np.concatenate((self.pos, self.neg), axis = 0)
        np.random.seed(1332)
        np.random.shuffle(self.pos_neg)

        csv_file_path = '../data/datasets/mod/train_otc.csv'
        fmt_list = ['%.0f', '%.0f', '%.0f'] + ['%.1f']
        np.savetxt(csv_file_path, np.hstack((np.arange(self.pos_neg.shape[0]).reshape(-1, 1), self.pos_neg)), delimiter=',', header=',source,target,weight', comments='', fmt=fmt_list)  
        
        self.train_edges = self.pos_neg[:,:-1]
        self.y_train = self.pos_neg[:,-1]
        
        self.test_pos = np.concatenate((self.test_positive_edges, self.test_positive_labels), axis = 1)
        self.test_neg = np.concatenate((self.test_negative_edges, self.test_negative_labels), axis = 1)
        self.test_pos_neg = np.concatenate((self.test_pos, self.test_neg), axis = 0)
        np.random.seed(2227)
        np.random.shuffle(self.test_pos_neg)

        csv_file_path = '..//data/datasets/mod/test_otc.csv'

        fmt_list = ['%.0f', '%.0f', '%.0f'] + ['%.1f']
        np.savetxt(csv_file_path, np.hstack((np.arange(self.test_pos_neg.shape[0]).reshape(-1, 1), self.test_pos_neg)), delimiter=',', header=',source,target,weight', comments='', fmt=fmt_list)
       
        self.test_edges = self.test_pos_neg[:,:-1]
        self.y_test = self.test_pos_neg[:,-1]
    

        one_hot = np.array(pd.get_dummies(list(np.array(self.y_train).ravel())), dtype=np.int8)
        self.y_train_one_hot = np.array(one_hot)
        one_hot3 = np.array(pd.get_dummies(list(np.array(self.y_test).ravel())), dtype=np.int8)
        self.y_test_one_hot = np.array(one_hot3)
        
        
        self.ecount = len(self.train_edges)
        self.train_edges = np.array(self.train_edges)
        self.num_labels = np.shape(self.y_train_one_hot)[1]

        self.X = setup_features(self.args)

        self.sorted_train_edges = []
        index = []
        for label in range(self.num_labels):
            label_index = [i for i in range(len(self.y_train_one_hot)) if self.y_train_one_hot[i][label] == 1]
            self.sorted_train_edges.append(self.train_edges[label_index,:])
            index.append(len(label_index))

        self.y = []
        for i in range(len(index)):
            self.y = np.append(self.y, [i]*index[i])
        self.y = np.append(self.y, [len(index)]*self.args.nolink_size*self.ecount)
        self.y = torch.from_numpy(self.y).type(torch.LongTensor).to(self.device)

        self.train_edges = torch.from_numpy(np.array(self.train_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        self.y_train = torch.from_numpy(np.array(self.y_train, dtype=np.float32)).type(torch.float).to(self.device)
        self.y_train_one_hot = torch.from_numpy(np.array(self.y_train_one_hot, dtype=np.float32)).type(torch.float).to(self.device)
        self.positive_edges = torch.from_numpy(np.array(self.positive_edges,
                                                        dtype=np.int64).T).type(torch.long).to(self.device)
        self.negative_edges = torch.from_numpy(np.array(self.negative_edges,
                                                        dtype=np.int64).T).type(torch.long).to(self.device)
        self.positive_labels = torch.from_numpy(np.array(self.positive_labels, dtype=np.float32)).type(torch.float).to(self.device)
        self.negative_labels= torch.from_numpy(np.array(self.negative_labels, dtype=np.float32)).type(torch.float).to(self.device)
        self.num_labels = torch.from_numpy(np.array(self.num_labels, dtype=np.int64)).type(torch.long).to(self.device)
        self.X = torch.from_numpy(self.X).to(self.device)


    def create_and_train_model(self):
        """
        Model training and scoring.
        """
        self.model = GraphConvolutionalNetwork(self.device, self.args, self.X, self.num_labels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.model.train()
        self.epochs = trange(self.args.epochs, desc="Loss")

        start_time = time.time()
        for epoch in self.epochs:

            self.optimizer.zero_grad()
            
            loss, _ = self.model(self.positive_edges, self.negative_edges, self.train_edges, self.y_train, self.positive_labels, self.negative_labels,self.sorted_train_edges,self.y)
            loss.backward()
            self.epochs.set_description("FineTrust (Loss=%g)" % round(loss.item(), 4))
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            self.logs["training_time"].append([epoch + 1, time.time() - start_time])

        end = time.time()
        print(f"train time: {end - start_time:.2f} seconds")
        self.save_model() 

    def save_model(self):
        """
        Saving the embedding and model weights.
        """
        print("\nEmbedding is saved.\n")
        loss, self.train_z = self.model(self.positive_edges, self.negative_edges,self.train_edges, self.y_train, self.positive_labels, self.negative_labels,self.sorted_train_edges,self.y) 
        train_z = self.train_z.cpu().detach().numpy()
        with open(self.args.embedding_path, 'w') as embed:
            for line in train_z:
                for item in line:
                    embed.write('%s ' % item)
                embed.write('\n')
    





