from model import *
from dataloader import *
from util import *
from pytorch_pretrained_bert.optimization import BertAdam
from myloss import *
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch

from pytorch_pretrained_bert.optimization import BertAdam
from myloss import *
import matplotlib.pyplot as plt
from transformers import get_scheduler
from torch.optim import AdamW
from torch.optim import AdamW
# from transformers import AdamW, get_scheduler
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import colors as mcolors
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from utils import util
import seaborn as sns
def set_seed(seed):
    random.seed(seed)  # Set Python random seed
    np.random.seed(seed)  # Set numpy random seed
    torch.manual_seed(seed)  # Set PyTorch seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set PyTorch seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmark
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
class PretrainModelManager:

    def __init__(self, args, data):

        self.model = BertForModel.from_pretrained(args.bert_model, cache_dir="", num_labels=data.num_labels)

        if args.freeze_bert_parameters:
            for name, param in self.model.bert.named_parameters():
                param.requires_grad = False
                if "encoder.layer.11" in name or "pooler" in name:
                    param.requires_grad = True

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.num_train_optimization_steps = len(data.train_dataloader) * args.num_train_epochs

        self.optimizer = self.get_optimizer(args)


        self.best_eval_score = 0
        self.best_evaluation_score = 0
        self.best_original_examples=None
        self.clusterLoss = clusterLoss(args, data)
        self.gb_centroids = None
        self.gb_radii = None
        self.gb_labels = None



    def euclidean_metric(self,a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        return torch.sqrt(torch.pow(a - b, 2).sum(2))


    def eval_by_prototype(self, args, data, known_prototypes, known_prototype_labels):



        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)  # 储存真实标签
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)  # 储存预测类别


        known_prototypes = known_prototypes.to(self.device)
        known_prototype_labels = known_prototype_labels.to(self.device)

        for batch in tqdm(data.eval_dataloader, desc="Eval by Prototype"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, label_noiseids, index = batch

            with torch.set_grad_enabled(False):

                features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)


                dists = torch.cdist(features, known_prototypes, p=2)
                pred_proto_idx = torch.argmin(dists, dim=1)
                pred_labels = known_prototype_labels[pred_proto_idx]

                total_labels = torch.cat((total_labels, label_ids))
                total_preds = torch.cat((total_preds, pred_labels))

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        return acc


    def eval(self, args, data):


        self.model.eval()
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)

        for batch in tqdm(data.eval_dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, label_noiseids, index = batch
            with torch.set_grad_enabled(False):
                _, logits = self.model(input_ids, segment_ids, input_mask,
                                       mode='eval')
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        return acc




    def compute_cluster_loss(self, clean_ind_label, ind_all_indice, ood_all_indice, accumulated_features,
                             accumulated_indexs):

        ind_centers = []
        for single_indice in ind_all_indice:
            find_indices = torch.tensor(single_indice, dtype=torch.long).to('cuda:0')
            find_positions = torch.nonzero(accumulated_indexs.unsqueeze(1) == find_indices, as_tuple=False)[:, 0]
            feature_for_center = accumulated_features[find_positions].to('cuda:0')
            center = feature_for_center.mean(axis=0)
            ind_centers.append(center)

        ood_centers = []
        for single_indice in ood_all_indice:
            find_indices = torch.tensor(single_indice, dtype=torch.long).to('cuda:0')
            find_positions = torch.nonzero(accumulated_indexs.unsqueeze(1) == find_indices, as_tuple=False)[:, 0]
            feature_for_center = accumulated_features[find_positions].to('cuda:0')
            center = feature_for_center.mean(axis=0)
            ood_centers.append(center)

        ind_centers = torch.stack(ind_centers) if ind_centers else torch.empty(0).to(
            'cuda:0')
        ood_centers = torch.stack(ood_centers) if ood_centers else torch.empty(0).to(
            'cuda:0')
        if ind_centers.size(0) > 0:
            pairwise_dist_clean = torch.cdist(ind_centers, ind_centers,p=2)


        if ood_centers.size(0) and ind_centers.size(0) > 0:
            pairwise_dist_ood = torch.cdist(ind_centers, ood_centers, p=2)

        inter_loss = 0.0
        intra_loss = 0.0
        ood_loss = 0.0


        if ind_centers.size(0) > 0:
            for i in range(ind_centers.size(0)):
                for j in range(ind_centers.size(0)):
                    if i == j:
                        continue
                    if clean_ind_label[i] == clean_ind_label[j]:
                        intra_loss += pairwise_dist_clean[i, j] ** 2


        if ind_centers.size(0) > 0:
            for i in range(ind_centers.size(0)):
                for j in range(ind_centers.size(0)):
                    if i == j:
                        continue
                    if clean_ind_label[i] != clean_ind_label[j]:
                        inter_loss += 1.0 / (pairwise_dist_clean[i, j] ** 2 + 1e-6)


        if ood_centers.size(0) and ind_centers.size(0)> 0:
            for i in range(ind_centers.size(0)):
                for j in range(ood_centers.size(0)):
                    ood_loss += 1.0 / (pairwise_dist_ood[i, j] ** 2 + 1e-6)


        num_clean_centers = ind_centers.size(0)
        num_ood_centers = ood_centers.size(0)
        if ind_centers.size(0) > 0:
            intra_loss = intra_loss / (num_clean_centers * (num_clean_centers - 0.9))
            inter_loss = inter_loss / (num_clean_centers * (num_clean_centers - 0.9))
        if ood_centers.size(0)>0 and ind_centers.size(0) > 0:
            ood_loss = ood_loss / (
                    num_clean_centers * num_ood_centers) if num_ood_centers > 0 else 0
        if ind_centers.size(0) > 0 and ood_centers.size(0)>0 :
            total_loss = intra_loss + inter_loss + ood_loss

        if ind_centers.size(0) ==0 and ood_centers.size(0) > 0:
            total_loss = 0
        if ind_centers.size(0) >0 and ood_centers.size(0) == 0:
            total_loss = intra_loss + inter_loss

        return intra_loss,inter_loss,ood_loss

    def prototype_softmax_loss(self, clean_ind_indice, clean_ind_label, known_protypes, known_protype_labels,
                               accumulated_features, accumulated_indexs):
        device = accumulated_features.device
        index_map = {int(idx): i for i, idx in enumerate(accumulated_indexs.cpu().numpy())}
        selected_positions = [index_map[i] for i in clean_ind_indice if i in index_map]
        selected_features = accumulated_features[selected_positions]  # (N, D)

        dists = torch.cdist(selected_features, known_protypes, p=2) ** 2  # (N, P)
        target_indices = []

        for i, label in enumerate(clean_ind_label):
            matching_indices = torch.where(known_protype_labels == label)[0]
            if matching_indices.numel() == 0:
                continue
            d_i = dists[i, matching_indices]
            nearest_index = matching_indices[torch.argmin(d_i)].item()
            target_indices.append(nearest_index)

        if len(target_indices) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        target_indices = torch.tensor(target_indices, dtype=torch.long, device=device)
        logits = -dists[:len(target_indices)]
        return F.cross_entropy(logits, target_indices, reduction='mean')

    def margin_prototype_loss(self, clean_ind_indices, clean_ind_labels, known_prototypes, known_prototype_labels,
                              accumulated_features, accumulated_indexs, margin=0.5):
        device = accumulated_features.device
        index_map = {int(idx): i for i, idx in enumerate(accumulated_indexs.cpu().numpy())}
        valid_features, valid_target_indices = [], []

        for i, label in enumerate(clean_ind_labels):
            if clean_ind_indices[i] not in index_map:
                continue
            pos = index_map[clean_ind_indices[i]]
            feature = accumulated_features[pos]
            matching_indices = torch.where(known_prototype_labels == label)[0]
            if matching_indices.numel() == 0:
                continue
            d_i = torch.cdist(feature.unsqueeze(0), known_prototypes, p=2).squeeze(0) ** 2
            nearest_index = matching_indices[torch.argmin(d_i[matching_indices])].item()
            valid_features.append(feature)
            valid_target_indices.append(nearest_index)

        if not valid_features:
            return torch.tensor(0.0, device=device, requires_grad=True)

        features = torch.stack(valid_features)
        target_indices = torch.tensor(valid_target_indices, dtype=torch.long, device=device)
        dists = torch.cdist(features, known_prototypes, p=2) ** 2
        pos_dists = dists[torch.arange(len(features)), target_indices]

        logits_list = []
        for i in range(len(features)):
            pos_logit = -pos_dists[i].unsqueeze(0)
            mask = torch.ones(dists.size(1), dtype=torch.bool, device=device)
            mask[target_indices[i]] = False
            neg_logits = -(dists[i, mask] + margin)
            logits_list.append(torch.cat([pos_logit, neg_logits]))

        logits = torch.stack(logits_list)
        return F.cross_entropy(logits, torch.zeros(len(features), dtype=torch.long, device=device), reduction='mean')

    def ood_repel_loss(self, ood_indices, accumulated_features, known_prototypes, accumulated_indexs, tau=1.0, K=5):
        device = accumulated_features.device
        index_map = {int(idx): i for i, idx in enumerate(accumulated_indexs.cpu().numpy())}
        valid_positions = [index_map[i] for i in ood_indices if i in index_map]
        if not valid_positions:
            return torch.tensor(0.0, device=device, requires_grad=True)

        ood_features = accumulated_features[valid_positions]
        dists = torch.cdist(ood_features, known_prototypes, p=2) ** 2
        topk_dists, _ = torch.topk(dists, K, dim=1, largest=False)
        repel = torch.exp((tau - topk_dists) / tau)
        return repel.mean()

    def train(self, args, data):
        wait = 0
        wait_ood_train=0
        best_model = None
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            if epoch < args.warm_train_epoch:
                for step, batch in enumerate(data.train_dataloader):
                    batch = tuple(t.to(self.device) for t in
                                  batch)
                    input_ids, input_mask, segment_ids, label_ids,label_noiseids,index= batch
                    batch_number = len(data.train_dataloader)
                    with ((torch.set_grad_enabled(True))):

                            loss1 = self.model(input_ids, segment_ids, input_mask, label_noiseids,mode="train")

                            self.optimizer.zero_grad()
                            loss1.backward()
                            self.optimizer.step()
                            tr_loss += loss1.item()
                            util.summary_writer.add_scalar("Loss/loss1", loss1.item(), step + epoch * batch_number)
                            nb_tr_examples += input_ids.size(0)
                            nb_tr_steps += 1

                loss = tr_loss / nb_tr_steps
                print('train_loss', loss)

                eval_score = self.eval(args, data)
                print('eval_score', eval_score)
                if eval_score > self.best_eval_score:
                    best_model = copy.deepcopy(self.model)
                    wait = 0
                    self.best_eval_score = eval_score
                else:
                    wait += 1
                    if wait >= args.wait_patient:
                        break
            else:
                self.draw4(args, data)
                memory_bank = []
                memory_bank_label = []
                memory_bank_noise_label = []
                memory_bank_index = []
                memory_bank_input_ids = []
                memory_bank_input_mask = []
                memory_bank_segment_ids = []


                for step, batch in enumerate(data.train_dataloader):
                    batch = tuple(t.to(self.device) for t in
                                  batch)
                    input_ids, input_mask, segment_ids, label_ids, label_noiseids, index = batch
                    batch_number = len(data.train_dataloader)
                    with (((torch.set_grad_enabled(True)))):

                        features = self.model(input_ids, segment_ids, input_mask, feature_ext="True")
                        memory_bank.append(features.cpu())
                        memory_bank_label.append(label_ids.cpu())
                        memory_bank_input_ids.append( input_ids.cpu())
                        memory_bank_input_mask.append(input_mask.cpu())
                        memory_bank_segment_ids.append(segment_ids.cpu())
                        memory_bank_noise_label.append(label_noiseids.cpu())
                        memory_bank_index.append(index.cpu())

                        if (step + 1) == batch_number:

                            accumulated_features = torch.cat(memory_bank, dim=0).to('cuda:0')
                            accumulated_labels = torch.cat(memory_bank_label, dim=0).to('cuda:0')
                            accumulated_noise_labels = torch.cat(memory_bank_noise_label, dim=0).to('cuda:0')
                            accumulated_indexs=torch.cat(memory_bank_index, dim=0).to('cuda:0')
                            memory_bank_input_ids=torch.cat(memory_bank_input_ids, dim=0).to('cuda:0')
                            memory_bank_input_mask=torch.cat(memory_bank_input_mask, dim=0).to('cuda:0')
                            memory_bank_segment_ids= torch.cat(memory_bank_segment_ids, dim=0).to('cuda:0')


                            clean_ind_indice, clean_ind_label,ood_indice,known_protypes,known_protype_labels,all_index,all_vaule_GIND,all_value_SOOD=self.clusterLoss.forward(args,accumulated_features,accumulated_labels,accumulated_noise_labels,accumulated_indexs,
                                                                                                               select=False)
                            plot_gind_sood_from_index_sampled(
                                all_index, all_vaule_GIND, all_value_SOOD,
                                accumulated_labels, accumulated_noise_labels, accumulated_indexs,
                                max_per_class=300
                            )
                            plot_gind_sood_from_index_sampled_with_distribution(
                                all_index, all_vaule_GIND, all_value_SOOD,
                                accumulated_labels, accumulated_noise_labels, accumulated_indexs,
                                max_per_class=300
                            )

                            known_protypes=torch.tensor(np.array(known_protypes), dtype=torch.float32).to(self.device)
                            clean_ind_label = torch.tensor(clean_ind_label)
                            known_protype_labels=torch.tensor(known_protype_labels)
                            prototype_softmax_loss=self.prototype_softmax_loss(clean_ind_indice, clean_ind_label,known_protypes,known_protype_labels,accumulated_features,
                             accumulated_indexs)
                            print("prototype_softmax_loss",prototype_softmax_loss)
                            margin_prototype_loss=self.margin_prototype_loss(clean_ind_indice, clean_ind_label,known_protypes,known_protype_labels,accumulated_features,
                             accumulated_indexs, margin=0.5)
                            print("margin_prototype_loss",margin_prototype_loss)
                            ood_repel_loss=self.ood_repel_loss(ood_indice, accumulated_features,known_protypes, accumulated_indexs, tau=6.0, K=5)
                            print("ood_repel_loss",ood_repel_loss)
                            total_loss=args.alpha*prototype_softmax_loss+ args.beta* margin_prototype_loss+  args.gamma*ood_repel_loss
                            print("total_loss",total_loss)



                            self.optimizer.zero_grad()
                            total_loss.backward()
                            self.optimizer.step()


                            util.summary_writer.add_scalar("Loss/prototype_softmax_loss", prototype_softmax_loss,
                                                           step + epoch * batch_number)
                            util.summary_writer.add_scalar("Loss/margin_prototype_loss", margin_prototype_loss,
                                                           step + epoch * batch_number)
                            util.summary_writer.add_scalar("Loss/ood_repel_loss",  ood_repel_loss,
                                                           step + epoch * batch_number)
                            util.summary_writer.add_scalar("Loss/total_loss", total_loss.item(), step + epoch * batch_number)
                            nb_tr_examples += input_ids.size(0)


                self.draw4(args, data)

                print('train_loss', total_loss.item())


                evaluation_score=self.eval_by_prototype(args, data,known_protypes,known_protype_labels)
                print('evaluation_score', evaluation_score)
                if evaluation_score > self.best_evaluation_score:
                    best_model = copy.deepcopy(self.model)
                    wait_ood_train = 0
                    self.best_evaluation_score = evaluation_score

                else:
                    wait_ood_train += 1
                    if  wait_ood_train >= args.wait_patient:
                        break

        self.model = best_model
        if args.save_model:
            self.save_model(args)
        self.draw4(args, data)

    def compute_classification_loss(self, data,features, labels, centroids, centroid_labels):
        if features.size(0) == 0:
            return torch.tensor(0.0).to(features.device)


        logits = torch.cdist(features, centroids, p=2)
        distances = torch.full((features.shape[0], len(data.known_label_list)), float('inf')).to(logits.device)


        for label in range(len(data.known_label_list)):
            class_mask = (centroid_labels == label)
            if class_mask.any():
                class_distances = logits[:, class_mask]
                distances[:, label] = class_distances.min(dim=1)[0]


        distances = F.normalize(distances, p=1, dim=1)

        probabilities = F.softmax(-distances, dim=1)

        true_probabilities = probabilities[torch.arange(probabilities.size(0)), labels]

        loss = -torch.log(true_probabilities).mean()
        return loss

    def calculate_granular_balls(self, args, data):

        for epoch in trange(int(1), desc="Epoch"):
            self.model.train()
            memory_bank = []
            memory_bank_label = []
            memory_bank_noise_label = []
            memory_bank_index = []

            for step, batch in enumerate(data.train_dataloader):
                batch = tuple(t.to(self.device) for t in
                              batch)
                input_ids, input_mask, segment_ids, label_ids,label_noiseids,index = batch
                batch_number = len(data.train_dataloader)

                # 获取特征时不计算梯度
                features = self.model(input_ids, segment_ids, input_mask, feature_ext="True")
                memory_bank.append(features.cpu())  # 将特征移到CPU以节省GPU内存
                memory_bank_label.append(label_ids.cpu())
                memory_bank_noise_label.append(label_noiseids.cpu())
                memory_bank_index.append(index.cpu())

                if (step + 1) == batch_number:
                    accumulated_features = torch.cat(memory_bank, dim=0).to('cuda:0')
                    accumulated_labels = torch.cat(memory_bank_label, dim=0).to('cuda:0')
                    accumulated_noise_labels = torch.cat(memory_bank_noise_label, dim=0).to('cuda:0')
                    accumulated_indexs = torch.cat(memory_bank_index, dim=0).to('cuda:0')

                    gb_for_test_centers, gb_for_test_radius, gb_for_test_label= self.clusterLoss.forward(args, accumulated_features, accumulated_labels, accumulated_noise_labels,accumulated_indexs,select=True)

                    import numpy as np

                    # 优化：先转成 np.array 提高效率
                    gb_for_test_centers = np.array(gb_for_test_centers)
                    gb_for_test_centers = torch.tensor(gb_for_test_centers, dtype=torch.float32).to(self.device)

                    gb_for_test_radius = np.array(gb_for_test_radius)
                    gb_for_test_radius = torch.tensor(gb_for_test_radius, dtype=torch.float32).to(self.device)


                    gb_for_test_label = torch.tensor([int(x) for x in gb_for_test_label], dtype=torch.long).to(
                        self.device)

        return gb_for_test_centers, gb_for_test_radius, gb_for_test_label


    def draw4(self, args, data):
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.colors as mcolors
        import torch
        import random

        self.model.train()

        memory_bank = []
        memory_bank_label = []
        memory_bank_noise_label = []

        for step, batch in enumerate(data.train_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, label_noiseids, index = batch

            with torch.no_grad():
                features = self.model(input_ids, segment_ids, input_mask, feature_ext="True")
            memory_bank.append(features.cpu())
            memory_bank_label.append(label_ids.cpu())
            memory_bank_noise_label.append(label_noiseids.cpu())

        # 拼接全部样本
        features = torch.cat(memory_bank, dim=0).numpy()
        true_labels = torch.cat(memory_bank_label, dim=0).numpy()
        noise_labels = torch.cat(memory_bank_noise_label, dim=0).numpy()

        # 设置随机种子以保证可复现
        np.random.seed(1)
        random.seed(1)

        # TSNE 降维（在所有样本上）
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(features)

        # 在降维后的样本中随机选择 30% 进行可视化
        num_samples = X_2d.shape[0]
        sample_size = int(num_samples * 0.3)
        random_indices = np.random.choice(num_samples, sample_size, replace=False)

        # 找出 OOD 标签 ID
        ood_label_candidates = set(true_labels) - set(noise_labels)
        assert len(ood_label_candidates) == 1, f"Expected exactly one OOD label, got {ood_label_candidates}"
        ood_label_id = list(ood_label_candidates)[0]

        # 已知类标签
        known_label_ids = sorted(list(set(true_labels) - {ood_label_id}))

        # 自动生成颜色（适用于任意类数）
        def generate_distinct_colors(n):
            return [mcolors.hsv_to_rgb((i / n, 0.5, 0.9)) for i in range(n)]

        colors_list = generate_distinct_colors(len(known_label_ids))
        label_to_color = {label_id: colors_list[i] for i, label_id in enumerate(known_label_ids)}

        # 绘图
        plt.figure(figsize=(8, 6))
        for i in random_indices:
            x, y = X_2d[i]
            t_label = true_labels[i]
            n_label = noise_labels[i]

            if t_label == ood_label_id:
                plt.scatter(x, y, c='gray', marker='x', s=100, alpha=1)
            elif t_label != n_label:
                color = label_to_color.get(t_label, 'black')
                plt.scatter(x, y, c=[color], marker='x', s=100, alpha=1)
            else:
                color = label_to_color.get(t_label, 'black')
                plt.scatter(x, y, edgecolors=[color], facecolors='none', marker='o', s=50, alpha=0.8)

        plt.title(
            f"TSNE Visualization: {args.dataset}, known_cls_ratio={args.known_cls_ratio}, ind_noise_ratio={args.ind_noise_ratio}, ood_type={args.ood_type}")
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")
        #plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.show()


    def get_optimizer(self, args):

        param_optimizer = list(self.model.named_parameters())  # 获取参数的名称和值的元组列表
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  # 不对偏置（bias）、LayerNormalization 层的偏置和权重应用权重衰减

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            # 需要应用权重衰减
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            # 不需要应用权重衰减
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,  # 通过 BertAdam 类创建了一个优化器对象 optimizer
                             lr=args.lr,
                             warmup=args.warmup_proportion,  # 预热比例
                             t_total=self.num_train_optimization_steps)  # 总训练步数
        return optimizer




    def save_model(self, args):

        if not os.path.exists(args.pretrain_dir):  # 如果目录 args.pretrain_dir 不存在，就执行后续的代码
            os.makedirs(args.pretrain_dir)  # 调用 os.makedirs(args.pretrain_dir)，代码会检查指定路径的目录是否存在。如果不存在，它将创建该目录以及必要的父级目录
        self.save_model = self.model.module if hasattr(self.model,
                                                       'module') else self.model  # hasattr(self.model, 'module')  用于检查对象 self.model 是否有一个名为 'module' 的属性。

        model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(args.pretrain_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)  # 将 self.save_model 的状态字典（即模型的权重参数）保存到 model_file
        with open(model_config_file, "w") as f:  # 使用 open 函数以写入模式打开 model_config_file 文件
            f.write(self.save_model.config.to_json_string())  # 使用 f.write 方法将模型的配置信息以 JSON 字符串的形式写入该文件







def plot_gind_sood_from_index_sampled(
    all_index, all_vaule_GIND, all_value_SOOD,
    accumulated_labels, accumulated_noise_labels, accumulated_index,
    max_per_class=300
):
    # 1. 构建 index → position 的映射（方便快速查找）
    index_to_position = {int(idx): i for i, idx in enumerate(accumulated_index.cpu().numpy())}

    # 2. 提取 true/noise labels
    true_labels = accumulated_labels.cpu().numpy()
    noise_labels = accumulated_noise_labels.cpu().numpy()

    # 3. 找出 OOD 类标签
    ood_label_candidates = set(true_labels) - set(noise_labels)
    assert len(ood_label_candidates) == 1, f"Expected exactly one OOD label, got {ood_label_candidates}"
    ood_label_id = list(ood_label_candidates)[0]

    # 4. 分类收集点
    clean_pts, ind_pts, ood_pts = [], [], []

    for i, raw_idx in enumerate(all_index):
        raw_idx = int(raw_idx)
        if raw_idx not in index_to_position:
            continue
        pos = index_to_position[raw_idx]
        t_label = true_labels[pos]
        n_label = noise_labels[pos]
        gind = all_vaule_GIND[i]
        sood = all_value_SOOD[i]
        if sood == 0:
            sood = random.uniform(0.01, 0.2)
        if gind == 0:
            gind = random.uniform(0.01, 0.2)

        if sood== 1:
            sood = random.uniform(0.9, 1)
        if gind == 1:
            gind = random.uniform(0.9, 1)



        if t_label == ood_label_id:
            ood_pts.append((gind, sood))
        elif t_label != n_label:
            ind_pts.append((gind, sood))
        else:
            clean_pts.append((gind, sood))

    # 5. 抽样
    random.seed(42)
    clean_pts = random.sample(clean_pts, min(len(clean_pts), max_per_class))
    ind_pts = random.sample(ind_pts, min(len(ind_pts), max_per_class))
    ood_pts = random.sample(ood_pts, min(len(ood_pts), max_per_class))

    # 6. 拆坐标并绘图
    def unzip_safe(pairs):
        return zip(*pairs) if pairs else ([], [])

    clean_x, clean_y = unzip_safe(clean_pts)
    ind_x, ind_y = unzip_safe(ind_pts)
    ood_x, ood_y = unzip_safe(ood_pts)

    plt.figure(figsize=(8, 6))
    plt.scatter(clean_x, clean_y, edgecolors='blue', facecolors='none', marker='o', s=50, label='Clean')
    plt.scatter(ind_x, ind_y, c='blue', marker='x', s=70, label='IND noise')
    plt.scatter(ood_x, ood_y, c='red', marker='x', s=70, label='OOD noise')

    plt.xlabel('GIND')
    plt.ylabel('SOOD')
    plt.title('GIND vs SOOD (Sampled Clean / IND / OOD)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()




def plot_gind_sood_from_index_sampled_with_distribution(
    all_index, all_vaule_GIND, all_value_SOOD,
    accumulated_labels, accumulated_noise_labels, accumulated_index,
    max_per_class=300
):

    index_to_position = {int(idx): i for i, idx in enumerate(accumulated_index.cpu().numpy())}


    true_labels = accumulated_labels.cpu().numpy()
    noise_labels = accumulated_noise_labels.cpu().numpy()


    ood_label_candidates = set(true_labels) - set(noise_labels)
    assert len(ood_label_candidates) == 1, f"Expected exactly one OOD label, got {ood_label_candidates}"
    ood_label_id = list(ood_label_candidates)[0]


    clean_pts, ind_pts, ood_pts = [], [], []
    noise_type_list = []
    gind_list, sood_list = [], []

    for i, raw_idx in enumerate(all_index):
        raw_idx = int(raw_idx)
        if raw_idx not in index_to_position:
            continue
        pos = index_to_position[raw_idx]
        t_label = true_labels[pos]
        n_label = noise_labels[pos]
        gind = all_vaule_GIND[i]
        sood = all_value_SOOD[i]


        if t_label == ood_label_id:
            ood_pts.append((gind, sood))
            noise_type_list.append(2)
        elif t_label != n_label:
            ind_pts.append((gind, sood))
            noise_type_list.append(1)
        else:
            clean_pts.append((gind, sood))
            noise_type_list.append(0)

        gind_list.append(gind)
        sood_list.append(sood)


    random.seed(42)
    clean_pts = random.sample(clean_pts, min(len(clean_pts), max_per_class))
    ind_pts = random.sample(ind_pts, min(len(ind_pts), max_per_class))
    ood_pts = random.sample(ood_pts, min(len(ood_pts), max_per_class))

    def unzip_safe(pairs):
        return zip(*pairs) if pairs else ([], [])

    clean_x, clean_y = unzip_safe(clean_pts)
    ind_x, ind_y = unzip_safe(ind_pts)
    ood_x, ood_y = unzip_safe(ood_pts)

    # 6. 画散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(clean_x, clean_y, edgecolors='blue', facecolors='none', marker='o', s=50, label='Clean')
    plt.scatter(ind_x, ind_y, c='blue', marker='x', s=70, label='IND noise')
    plt.scatter(ood_x, ood_y, c='red', marker='x', s=70, label='OOD noise')
    plt.xlabel('GIND')
    plt.ylabel('SOOD')
    plt.title('GIND vs SOOD (Sampled Clean / IND / OOD)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 7. 绘制分布图
    gind_clean = [g for g, t in zip(gind_list, noise_type_list) if t == 0]
    gind_ind   = [g for g, t in zip(gind_list, noise_type_list) if t == 1]
    gind_ood   = [g for g, t in zip(gind_list, noise_type_list) if t == 2]

    sood_clean = [s for s, t in zip(sood_list, noise_type_list) if t == 0]
    sood_ind   = [s for s, t in zip(sood_list, noise_type_list) if t == 1]
    sood_ood   = [s for s, t in zip(sood_list, noise_type_list) if t == 2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.kdeplot(gind_clean, ax=axes[0], label='Clean', color='green')
    sns.kdeplot(gind_ind, ax=axes[0], label='IND noise', color='blue')
    sns.kdeplot(gind_ood, ax=axes[0], label='OOD noise', color='red')
    axes[0].set_title('GIND Distribution')
    axes[0].set_xlabel('GIND')
    axes[0].legend()

    sns.kdeplot(sood_clean, ax=axes[1], label='Clean', color='green')
    sns.kdeplot(sood_ind, ax=axes[1], label='IND noise', color='blue')
    sns.kdeplot(sood_ood, ax=axes[1], label='OOD noise', color='red')
    axes[1].set_title('SOOD Distribution')
    axes[1].set_xlabel('SOOD')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
