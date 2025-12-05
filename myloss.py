from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from cluster import gbcluster
from dataloader import*
from sklearn.metrics import pairwise_distances_argmin_min
from init_parameter import *
def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)

    b = b.unsqueeze(0).expand(n, m, -1)
    logits = ((a - b) ** 2).sum(dim=2)
    return logits


class MarginLoss(nn.Module):
    def __init__(self, num_class, size_average=True):
        super(MarginLoss, self).__init__()
        self.num_class = num_class
        self.size_average = size_average

    def forward(self, classes, labels):
        labels = F.one_hot(labels, self.num_class).float()
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        loss = labels * left + 0.5 * (1 - labels) * right
        loss = loss.sum(dim=-1)
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class clusterLoss(nn.Module):

    def __init__(self, args,data):

        super(clusterLoss, self).__init__()
        self.num_labels = data.num_labels
        self.feat_dim = args.feat_dim

        self.gbcluster=gbcluster(args,data)
        self.gb_centroids = None
        self.gb_radii = None
        self.gb_labels = None
        self.gb_result = None
        self.gb_purity=None


    def forward(self, args, features, labels, noise_labels,index,select=True):

        if select == True:
            gb_for_test_centers, gb_for_test_radius, gb_for_test_label= self.gbcluster.forward(args, features, labels,
                                                                                                noise_labels, index,
                                                                                                select)
            return gb_for_test_centers, gb_for_test_radius, gb_for_test_label


        else:
            clean_ind_indice, clean_ind_label,ood_indice,known_protypes,known_protype_labels,all_index, all_vaule_GIND,all_value_SOOD= self.gbcluster.forward(args, features, labels,noise_labels, index,select)

            if isinstance(clean_ind_indice, (list, np.ndarray)):
                clean_ind_indice = [int(x) for x in clean_ind_indice]
            else:
                clean_ind_indice = [int(clean_ind_indice)]

            if isinstance(ood_indice, (list, np.ndarray)):
                ood_indice = [int(x) for x in ood_indice]
            else:
                ood_indice = [int(ood_indice)]
            if isinstance(all_index, (list, np.ndarray)):
                all_index = [int(x) for x in all_index]
            else:
                all_index = [int(all_index)]


            return clean_ind_indice, clean_ind_label,ood_indice,known_protypes,known_protype_labels,all_index,all_vaule_GIND,all_value_SOOD






    def compute_classification_loss(self, features, labels, centroids, centroid_labels):
        if features.size(0) == 0:
            return torch.tensor(0.0).to(features.device)


        logits = torch.cdist(features, centroids, p=2)
        distances = torch.full((features.shape[0], self.num_labels), float('inf')).to(logits.device)


        for label in range(self.num_labels):
            class_mask = (centroid_labels == label)
            if class_mask.any():
                class_distances = logits[:, class_mask]
                distances[:, label] = class_distances.min(dim=1)[0]


        distances = F.normalize(distances, p=1, dim=1)

        probabilities = F.softmax(-distances, dim=1)

        true_probabilities = probabilities[torch.arange(probabilities.size(0)), labels]

        loss = -torch.log(true_probabilities).mean()
        return loss


