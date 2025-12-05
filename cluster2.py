import ast
import math
import time
import pandas as pd
import numpy as np
import numpy as np
from sklearn.metrics import pairwise_distances
import pandas as pd
import random
import csv
import torch
import numpy
import scipy.io
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
# from config import args
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import itertools
# import resnet
from GranularBallGeneration import *

# 1.输入数据data
# 2.打印绘制原始数据
# 3.判断粒球的纯度
# 4.纯度不满足要求，k-means划分粒球
# 5.绘制每个粒球的数据点
# 6.计算粒球均值，得到粒球中心和半径，绘制粒球


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import itertools

def set_seed(seed):
    random.seed(seed)  # Set Python random seed
    np.random.seed(seed)  # Set numpy random seed
    torch.manual_seed(seed)  # Set PyTorch seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set PyTorch seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmark
class Inputexamples(object):
    """A single set of features of data."""

    def __init__(self, numbers,label,centers,radius,mean_radius,purity,result,GIND,SOOD,index,noise_label):
        self.numbers = numbers
        self.label=label
        self.centers = centers
        self.radius = radius
        self.mean_radius=mean_radius
        self.purity = purity
        self.result = result
        self.GIND =GIND
        self.SOOD =SOOD
        self.index=index
        self.noise_label=noise_label

def calculate_center(data):
    return np.mean(data, axis=0)

def calculate_radius(data, center):
    return np.max(np.sqrt(np.sum((data - center) ** 2, axis=1)))





# 判断粒球的标签和纯度
def get_label_and_purity(gb):
    # 分离不同标签数据

    len_label = numpy.unique(gb[:, 1], axis=0)
    # print("len_label\n", len_label)  # 球内所有样本标签（不含重复)

    if len(len_label) == 1:  # 若球内只有一类标签样本，则纯度为1，将该标签定为球的标签
        purity = 1.0
        label = len_label[0]
    else:
        # 矩阵的行数
        num = gb.shape[0]  # 球内样本数
        gb_label_temp = {}
        for label in len_label.tolist():
            # 分离不同标签数据
            gb_label_temp[sum(gb[:, 1] == label)] = label
        # print("分离\n", gb_label_temp)  # dic{该标签对应样本数：标签类别}
        # 粒球中最多的一类数据占整个的比例
        try:
            max_label = max(gb_label_temp.keys())
        except:
            print("+++++++++++++++++++++++++++++++")
            print(gb_label_temp.keys())
            print(gb)
            print("******************************")
            exit()
        purity = max_label / num if num else 1.0  # pur为球内同一标签对应最多样本的类的样本数/球内总样本数

        label = gb_label_temp[max_label]  # 对应样本最多的一类定为球标签
    # print(label)
    # 标签、纯度
    return label, purity

def get_label_and_purity2(gb):
    # 分离不同标签数据

    len_label = numpy.unique(gb[:, 1], axis=0)
    # print("len_label\n", len_label)  # 球内所有样本标签（不含重复)

    if len(len_label) == 1:  # 若球内只有一类标签样本，则纯度为1，将该标签定为球的标签
        purity = 1.0
        label = len_label[0]
    else:
        # 矩阵的行数
        num = gb.shape[0]  # 球内样本数
        gb_label_temp = {}
        for label in len_label.tolist():
            # 分离不同标签数据
            gb_label_temp[sum(gb[:, 1] == label)] = label
        # print("分离\n", gb_label_temp)  # dic{该标签对应样本数：标签类别}
        # 粒球中最多的一类数据占整个的比例
        try:
            max_label = max(gb_label_temp.keys())
        except:
            print("+++++++++++++++++++++++++++++++")
            print(gb_label_temp.keys())
            print(gb)
            print("******************************")
            exit()
        purity = max_label / num if num else 1.0  # pur为球内同一标签对应最多样本的类的样本数/球内总样本数

        label = gb_label_temp[max_label]  # 对应样本最多的一类定为球标签
    # print(label)
    # 标签、纯度
    #return label, purity,len_label
    return label, purity

# 返回粒球中心和半径
def calculate_center_and_radius(gb,lab):
    sample_label=gb[:, 1]

    data_no_label = gb[:, 2:]  # 第3列往后的所有数据
    sample_indices = gb[:, 0]
    # print("data no label\n",data_no_label)
    center = data_no_label.mean(axis=0)  # 同一列在所有行之间求平均
    # print("center:\n", center)
    distances = np.linalg.norm(data_no_label - center, axis=1)
    radius = numpy.mean((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    valid_indices = sample_indices[distances < radius]  # 选出满足半径条件的样本索引
    valid_sample_label = sample_label[distances < radius]  # 同样条件下选出对应的标签

    valid_indices_clean = valid_indices[valid_sample_label == lab].tolist()
    valid_indices_IND = valid_indices[valid_sample_label != lab].tolist()
    valid_indices = valid_indices.tolist()

    return center, radius,valid_indices,sample_indices,valid_indices_clean,valid_indices_IND

def max_calculate_center_and_radius(gb):
    data_no_label = gb[:, 1:]  # 第2列往后的所有数据
    sample_indices=gb[:, 0]
    # print("data no label\n",data_no_label)
    center = data_no_label.mean(axis=0)  # 同一列在所有行之间求平均
    distances = np.linalg.norm(data_no_label - center, axis=1)
    radius = numpy.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    valid_indices = sample_indices[distances < radius]
    return center, radius,valid_indices





# def splits(args, gb_dict, select):
#     i = 0
#     keys = list(gb_dict.keys())  # 获取所有键的列表
#     while i < len(keys):  # 使用 i 来控制遍历，确保不会超出索引范围
#         key = keys[i]  # 当前处理的键
#         gb_dict_single = {key: gb_dict[key]}  # 提取当前键值对
#         gb = gb_dict_single[key][0]
#
#         if gb.shape[0] == 0:  # 如果粒球为空，跳过并继续处理下一个键
#             print(gb.shape)
#             i += 1
#             continue
#         else:
#             if isinstance(gb, torch.Tensor):
#                 gb = gb.cpu().numpy()  # 如果是Tensor类型，转为numpy数组
#             gb_dict_single[key][0] = gb  # 更新粒球的数据
#
#             distances = gb_dict_single[key][1]
#             if isinstance(distances, np.ndarray):
#                 distances = distances.tolist()  # 如果距离是NumPy数组，转为列表
#             gb_dict_single[key][1] = distances  # 更新距离
#
#             label, p = get_label_and_purity(gb)  # 获取粒球的标签和纯度
#             number_of_samples_in_ball = len(gb_dict_single[key][0])  # 获取粒球中样本的数量
#
#             # 根据select参数选择不同的纯度和最小样本数阈值
#             if select == False:
#                 a_purity = args.purity_train
#                 a_min_ball = args.min_ball_train
#             else:
#                 a_purity = args.purity_get_ball
#                 a_min_ball = args.min_ball_get_ball
#
#             # 如果满足条件，进行粒球的分割
#             if p < a_purity and number_of_samples_in_ball > a_min_ball:
#                 gb_dict_new = splits_ball(args, gb_dict_single).copy()  # 调用分割函数
#                 if len(gb_dict_new) > 1:
#                     gb_dict.pop(key)  # 删除旧的粒球
#                     gb_dict.update(gb_dict_new)  # 更新粒球字典
#                     keys.remove(key)  # 移除已处理的键
#                     keys.extend(gb_dict_new.keys())  # 添加新分割出的键
#                 else:
#                     if len(gb_dict_new) == 0 or key not in gb_dict_new:
#                         print(f"Warning: splits_ball returned an empty or invalid result for key {key}")
#                 # 分割处理完后，继续处理下一个键
#                 i += 1
#             else:
#                 i += 1  # 如果不满足分割条件，继续处理下一个键
#
#     return gb_dict

def splits(args, gb_dict, select):
    # 将 gb_dict 中所有样本拼接成一个 ndarray
    all_data = []
    for key in gb_dict:
        ball_data = gb_dict[key][0]
        if isinstance(ball_data, torch.Tensor):
            ball_data = ball_data.cpu().numpy()
        all_data.append(ball_data)
    data = np.vstack(all_data)
#data:index+noise_label+feature :1+1+768
    # 执行无监督粒球聚类
    gb_clusters = GBC(args,data)  # 返回多个粒球，每个是一个 ndarray

    # 构造新格式的 gb_dict_final


    return  gb_clusters


# 计算距离
def calculate_distances(data, p):  #原始的距离计算公式
    if isinstance(data, torch.Tensor) and isinstance(p, torch.Tensor):
        dis = (data - p).clone().detach() ** 2
        dis = dis.cpu().numpy()
    else:
        dis = (data - p) ** 2
    dis_top10 = np.sort(dis)[-10:]

    return 0.6 * np.sqrt(dis).sum() + 0.4 * np.sqrt(dis_top10).sum()
def calculate_distances2(data, p):
    if isinstance(data, torch.Tensor) and isinstance(p, torch.Tensor):
        dis = (data - p).clone().detach() ** 2
        dis = dis.cpu().numpy()
    else:
        dis = (data - p) ** 2


    return np.sqrt(dis.sum())


def splits_ball(args,gb_dict):
    # {center: [gb, distances]}
    center = []
    distances_other_class = []  # 粒球到异类点的距离
    balls = []  # 聚类后的label
    gb_dis_class = []  # 不同标签数据的距离
    center_other_class = []  # 与当前粒球标签不同的类
    center_distances = []  # 新距离
    ball_list = {}  # 最后要返回的字典，键：中心点，值：粒球 + 到中心的距离
    distances_other = []
    distances_other_temp = []

    centers_dict = []  # 中心list
    gbs_dict = []  # 粒球数据list
    distances_dict = []  # 距离list

    # 取出字典中的数据:center,gb,distances
    # 取字典数据，包括键值
    gb_dict_temp = gb_dict.popitem()
    for center_split in gb_dict_temp[0].split('_'):  # 将字典中原来是粒球质心分割，表示成正常的数值形式
        try:
            center.append(float(eval(center_split.strip())))
        except:
            center.append(float(center_split.strip()))
    center = np.array(center)  # 转为array
    centers_dict.append(center)  # 老中心加入中心list
    gb = gb_dict_temp[1][0] # 取出粒球数据
    distances = gb_dict_temp[1][1]  # 取出到老中心的距离


    # 分离不同标签数据的距离
    len_label = numpy.unique(gb[:, 1], axis=0)
    # print(len_label)
    for label in len_label.tolist():
        # 分离不同标签距离
        gb_dis_temp = []
        for i in range(0, len(distances)):
            if gb[i, 1] == label:
                gb_dis_temp.append(distances[i])
        if len(gb_dis_temp) > 0:
            gb_dis_class.append(gb_dis_temp)  # gb_dis_class 存了4个list，每个list是不同标签的样本到质心的距离

    if len(len_label)==1:
        for i in range(0, len(gb_dis_class)):
            # 随机异类点
            set_seed(args.seed)
            ran = random.randint(0, len(gb_dis_class[i]) - 1)
            center_other_temp = gb[distances.index(gb_dis_class[i][ran])]  # 随机点对应的标签和向量 # 随机点对应的标签和向量
            center_other_class.append(center_other_temp)
    else:
    # 取新中心
        for i in range(0, len(gb_dis_class)):

            # 随机异类点
            set_seed(args.seed)
            ran = random.randint(0, len(gb_dis_class[i]) - 1)
            center_other_temp = gb[distances.index(gb_dis_class[i][ran])]  # 随机点对应的标签和向量 # 随机点对应的标签和向量

            if center[1] != center_other_temp[1]:  # 判断新的粒球中心样本和旧类是否是相同的标签
                center_other_class.append(center_other_temp)

    centers_dict.extend(center_other_class)  # 在旧中心的基础上扩展了新的中心 一个旧的中心加上3个新的中心样本（中心样本不代表就是中心）


    distances_other_class.append(distances)  # 所有样本到原来质心的距离
    # 计算到每个新中心的距离
    for center_other in center_other_class:
        balls = []  # 聚类后的label
        distances_other = []
        for feature in gb:
            # 欧拉距离
            distances_other.append(calculate_distances(feature[2:], center_other[2:]))
        # 新中心list

        distances_other_temp.append(distances_other)  # 临时存放到每个新中心的距离
        distances_other_class.append(distances_other)


    # 某一个数据到原中心和新中心的距离，取最小以分类
    for i in range(len(distances)):
        distances_temp = []
        distances_temp.append(distances[i])
        for distances_other in distances_other_temp:
            distances_temp.append(distances_other[i])

        classification = distances_temp.index(min(distances_temp))  # 0:老中心；1,2...：新中心
        balls.append(classification)
    # 聚类情况
    balls_array = np.array(balls)


    # 根据聚类情况，分配数据
    for i in range(0, len(centers_dict)):
        gbs_dict.append(gb[balls_array == i, :])


    # 分配新距离
    i = 0
    for j in range(len(centers_dict)):
        distances_dict.append([])

    for label in balls:
        distances_dict[label].append(distances_other_class[label][i])
        i += 1


    # 打包成字典
    for i in range(len(centers_dict)):
        gb_dict_key = str(centers_dict[i][0])
        for j in range(1, len(centers_dict[i])):
            gb_dict_key += '_' + str(centers_dict[i][j])
        gb_dict_value = [gbs_dict[i], distances_dict[i]]  # 粒球 + 到中心的距离
        ball_list[gb_dict_key] = gb_dict_value

    return ball_list

import numpy as np

import numpy as np

import numpy as np

# def compute_sample_GIND(sample_distances_to_center, gb_purity, gb_radius, epsilon=1e-6):
#     """
#     计算每个样本的 GIND 值，并进行 sigmoid 归一化（用于全局区分 IND/OOD 噪声）。
#
#     参数:
#         sample_distances_to_center: ndarray, 每个样本到粒球质心的 L2 距离
#         gb_purity: float, 粒球的纯度 (0~1)
#         gb_radius: float, 粒球半径
#         epsilon: float, 防止除 0
#
#     返回:
#         gind_raw: ndarray, 原始 GIND 值
#         gind_norm: ndarray, 归一化后的 GIND 值
#     """
#     gb_radius = max(gb_radius, epsilon)
#     purity_factor = gb_purity ** 2  # 平滑抑制低纯度
#     gind_raw = purity_factor * np.exp(-sample_distances_to_center / gb_radius)
#
#     # Sigmoid 映射：适度拉伸压缩
#     #gind_norm = 1 / (1 + np.exp(-2 * gind_raw + 3))
#
#     return  gind_raw
def compute_sample_GIND(distances, purity, radius, epsilon=1e-6):
    """
    计算 GIND 值，并在粒球内部进行归一化。

    参数:
        distances: ndarray, 每个样本到中心的 L2 距离
        purity: float, 粒球的纯度 (0~1)
        radius: float, 粒球半径
    返回:
        gind_norm: ndarray, 归一化后的 GIND 值 (粒球内 min-max)
    """
    radius = max(radius, epsilon)
    gind_raw = purity * np.exp(-distances / radius)

    # 粒球内归一化（避免出现除以0）
    min_val = np.min(gind_raw)
    max_val = np.max(gind_raw)
    if max_val - min_val < epsilon:
        gind_norm = np.ones_like(gind_raw) * purity  # 所有值一样，赋为纯度
    else:
        gind_norm = (gind_raw - min_val) / (max_val - min_val)

    return gind_raw




def compute_SOOD(X, all_gb_centers, all_gb_radius, all_gb_sizes, K=5, epsilon=1e-6):
    """
    批量计算多个样本的 SDI 值并进行 tanh 归一化。

    返回:
        sdi_raw_array: ndarray, shape=(N,)
        sdi_norm_array: ndarray, shape=(N,)
    """
    N = X.shape[0]
    sdi_raw_array = np.zeros(N)
    sdi_norm_array = np.zeros(N)
    N_max = max(all_gb_sizes)

    distance_matrix = np.array([[np.linalg.norm(x - c) for c in all_gb_centers] for x in X])

    for i in range(N):
        distances = distance_matrix[i]
        top_k_idx = np.argsort(distances)[:K]

        sdi_raw = 0.0
        for idx in top_k_idx:
            R_k = max(all_gb_radius[idx], epsilon)
            N_Bk = all_gb_sizes[idx]
            normalized_distance = distances[idx] / R_k
            #trust_weight = 1 - (N_Bk / N_max)
            #sdi_raw += normalized_distance * trust_weight
            sdi_raw += normalized_distance
        sdi_raw /= K
        sdi_raw_array[i] = sdi_raw
        sdi_norm_array[i] = np.tanh(0.5 * sdi_raw)

    return sdi_raw_array

from sklearn.metrics import pairwise_distances
import numpy as np

import numpy as np
from sklearn.metrics import pairwise_distances

def compute_gb_ood(all_centers, gb_label, gb_purity, M=5, K=5, purity_threshold=0.5):
    all_centers = np.array(all_centers)
    gb_label = np.array(gb_label)
    gb_purity = np.array(gb_purity)
    N = len(all_centers)
    gb_ood_scores = np.zeros(N)
    distance_matrix = pairwise_distances(all_centers)

    for j in range(N):
        label_j = gb_label[j]

        # Step 1: 计算 intra-class 平均距离（排除自己）
        same_label_mask = (gb_label == label_j) & (gb_purity > purity_threshold) & (np.arange(N) != j)
        same_label_indices = np.where(same_label_mask)[0]
        if len(same_label_indices) > 0:
            same_distances = distance_matrix[j][same_label_indices]
            top_m = min(M, len(same_label_indices))
            D_intra = np.mean(np.sort(same_distances)[:top_m])
        else:
            D_intra = 0.0

        # Step 2: 邻居标签不一致率
        nearest_indices = np.argsort(distance_matrix[j])[1:K+1]
        R_inter = np.mean(gb_label[nearest_indices] != label_j)

        gb_ood_scores[j] = D_intra * R_inter

    # Step 3: Sigmoid 归一化
    sood_scores = np.array(gb_ood_scores)
    min_score = np.min(sood_scores)
    max_score = np.max(sood_scores)
    sood_norm = (sood_scores - min_score) / (max_score - min_score + 1e-6)
    return sood_norm






def main(args,data,select):  # +pur
#data:index+noise_label+feature 1+1+768
    # 初始随机中心
    set_seed(args.seed)
    center_init = data[random.randint(0, len(data) - 1), :]  # 任选一行，也就是某个样本作为初始中心
    distance_init = np.array([calculate_distances(feature[2:], center_init[2:]) for feature in data]) #计算所有样本距离初始中心（后64维向量）的欧拉距离：差的平方和的开方

    # 封装成字典
    gb_dict = {}
    gb_dict_key = str(center_init.tolist()[1])  # 质心的标签+向量
    for j in range(2, len(center_init)):
        gb_dict_key += '_' + str(center_init.tolist()[j])

    gb_dict_value = [data, distance_init]  # 所有样本1+1+768+该样本到中心的距离3333
    gb_dict[gb_dict_key] = gb_dict_value  # gb_dict 是一个字典，字典里面包含了其对应的 粒球（质心+标签）+粒球中的样本（标签+向量）+样本到质心的距离。


    # 分类划分
    gb_dict = splits(args,gb_dict,select)
    examples=[]
    all_centers=[]
    all_radii=[]
    all_sizes=[]


    gb_label = []
    gb_purity  = []
    for i in range(0, len(gb_dict)):
        gb = gb_dict[i]
        gb_center = gb.center
        all_centers.append(gb_center)
        gb_radius = gb.radius
        all_radii.append(gb_radius)
        gb_num = gb.num
        all_sizes.append(gb_num)
        gb_label_each, gb_purity_each = get_label_and_purity2(gb.data)

        gb_label.append(gb_label_each)
        gb_purity.append(gb_purity_each)
    GBOOD_scores= compute_gb_ood(all_centers, gb_label, gb_purity, M=5, K=5)

    for i in range(0, len(gb_dict)):
        gb = gb_dict[i]
        gb_data=gb.data
        gb_center=gb.center
        gb_radius= gb.radius
        gb_num = gb.num
        gb_label_each, gb_purity_each= get_label_and_purity2(gb_data)

        sample_index=gb_data[:, 0]
        sample_noise_label = gb_data[:, 1]
        sample_data_no_label = gb_data[:, 2:]
        sample_distances_to_center = np.linalg.norm(sample_data_no_label - gb_center, axis=1)
        gb_mean_radius = numpy.mean(sample_distances_to_center)
        sample_GIND = compute_sample_GIND(sample_distances_to_center, gb_purity_each, gb_radius)
        sample_SOOD = compute_SOOD(sample_data_no_label, all_centers, all_radii, all_sizes)
        GBOOD = GBOOD_scores[i]

        examples.append(
            Inputexamples(numbers=gb_num,
                  label=gb_label_each,
                  centers=gb_center,
                  radius=gb_radius,
                  mean_radius=gb_mean_radius,
                  purity=gb_purity_each,
                  result=gb_data,
                  GIND=sample_GIND,
                  SOOD=GBOOD,
                  index=sample_index,
                  noise_label=sample_noise_label
                          ))


    save_path = "E:\code_hh\PNGB\BNO_GB\outputs\granular_ball_summary.csv"
    with open(save_path, "w", newline="", encoding="utf-8") as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(["Label", "Num_Samples", "Mean_Radius", "Purity", "GIND", "SOOD"])
        for example in examples:
            label = example.label
            numbers = example.numbers
            mean_radius = example.mean_radius
            purity = example.purity
            GIND = example.GIND
            SOOD = example.SOOD
            writer.writerow([label, numbers,mean_radius, purity, GIND, SOOD])



    return examples




def summarize_granular_balls(examples):

    summary = []
    for i, ball in enumerate(examples):
        summary.append({
            "Ball_ID": i,
            "Label": ball.label,
            "Num_Samples": ball.numbers,
            "Radius": round(ball.radius, 4),
            "Purity": round(ball.purity, 4),
            "Avg_GIND": round(np.mean(ball.GIND), 4) if hasattr(ball, 'GIND') else None,
            "Avg_SOOD": round(np.mean(ball.SOOD), 4) if hasattr(ball, 'SOOD') else None
        })

    df_summary = pd.DataFrame(summary)
    return df_summary






