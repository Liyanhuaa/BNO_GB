from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import math
import numpy as np
from init_parameter import *
from dataloader import *
import torch.distributed as dist
import ast
import time
import pandas as pd
import random
import scipy.io
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
from scipy import stats
from dataloader import *
from init_parameter import *
import cluster2 as new_GBNR
def set_seed(seed):
    random.seed(seed)  # Set Python random seed
    np.random.seed(seed)  # Set numpy random seed
    torch.manual_seed(seed)  # Set PyTorch seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set PyTorch seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmark

class gbcluster(nn.Module):

    def __init__(self,args,data):
        super(gbcluster, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, args, features, labels,noise_labels,index,select):
        if select == False:
            a_purity = args.purity_train
        else:
            a_purity = args.purity_get_ball

        noise_label_features = torch.cat((noise_labels.reshape(-1, 1), features), dim=1)
        index_label_features = torch.cat((index.reshape(-1, 1), noise_label_features), dim=1)
        out = torch.cat((labels.reshape(-1, 1),index_label_features), dim=1)
        pur_tensor = torch.Tensor([[a_purity]] * out.size(0))
        out = torch.cat((pur_tensor.to(self.device), out), dim=1)
        examples= GBNR.apply(args,out.to(self.device),select)



        if  select==True:

            gb_for_test_centers = []
            gb_for_test_label = []
            gb_for_test_radius = []
            for example in examples:
                if example.purity>args.purity_select_ball and example.numbers>args.min_ball_select_ball:
                    gb_for_test_centers.append(example.centers)
                    gb_for_test_label.append(example.label)
                    gb_for_test_radius.append(example.mean_radius)
            return gb_for_test_centers, gb_for_test_radius, gb_for_test_label



        else:
            clean_ind_indice = []
            clean_ind_label = []
            ood_indice = []

            known_protypes = []
            known_protype_labels = []


            GIND_p = args.GIND_p
            SOOD_p = args.SOOD_p

            all_index=[]
            all_vaule_GIND=[]
            all_value_SOOD=[]

            for example in examples:
                n = 0
                for i in range(len(example.index)):
                    all_index.append(example.index[i])
                    all_vaule_GIND.append(example.GIND[i])
                    all_value_SOOD.append(example.SOOD)
                    if example.GIND[i]> GIND_p and example.SOOD<SOOD_p and example.numbers>2: #干净和ind噪声example.SOOD[i]
                        clean_ind_indice.append(example.index[i])
                        clean_ind_label.append(example.label)

                        n+=1
                    if example.GIND[i]< GIND_p and example.SOOD>SOOD_p: #干净和ind噪声
                        ood_indice.append(example.index[i])



                if example.purity>args.purity_select_ball_in_train and example.numbers>args.min_ball_select_ball_in_train:
                    known_protypes.append(example.centers)
                    known_protype_labels.append(example.label)

            return clean_ind_indice, clean_ind_label,ood_indice,known_protypes,known_protype_labels,all_index, all_vaule_GIND,all_value_SOOD



def calculate_distances(center, p):
    return ((center - p) ** 2).sum(axis=0) ** 0.5


class GBNR(torch.autograd.Function):
    @staticmethod
    def forward(self,args, input_,select):

        self.batch_size = input_.size(0)
        input_main = input_[:, 2:]
        self.input = input_[:, 4:]
        self.res = input_[:, 1:2]
        self.index = input_[:, 2:3]
        self.noise_labels = input_[:, 3:4]
        pur = input_[:, 0].cpu().numpy().tolist()[0]

        self.flag = 0
        examples = new_GBNR.main(args,input_main,select)
        print("example.shape",len(examples))
        return  examples

    @staticmethod
    def backward(self, output_grad, input, index, id, _):

        result = np.zeros([self.batch_size, 154], dtype='float64')  # +4

        for i in range(output_grad.size(0)):

            for a in self.balls[i]:
                input_np = np.array(self.input)
                a_np = np.array(a[1:])


                if input_np.shape[1:] == a_np.shape:
                    mask = (input_np == a_np).all(axis=1)
                    if mask.any():
                        result[mask, 4:] = output_grad[i, :].cpu().numpy()

        return torch.Tensor(result)


