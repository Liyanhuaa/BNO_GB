from init_parameter import *
from dataloader import *
from pretrain import *


import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
def set_seed(seed):
    random.seed(seed)  # Set Python random seed
    np.random.seed(seed)  # Set numpy random seed
    torch.manual_seed(seed)  # Set PyTorch seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set PyTorch seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmark

class ModelManager:

    def __init__(self, args, data,pretrained_model=None):

        self.model = pretrained_model
        if self.model is None:
            self.model = BertForModel.from_pretrained(args.bert_model, cache_dir = "", num_labels = data.num_labels)
            self.restore_model(args)

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id     
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.best_eval_score = 0
        self.delta = None
        self.delta_points = []
        self.centroids = None

        self.test_results = None
        self.predictions = None
        self.true_labels = None
        self.bset_gb_for_select_for_test_centers=None
        self.bset_gb_for_select_for_test_radius =None
        self.bset_gb_for_select_for_test_label=None



    def euclidean_metric(self,a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        return torch.sqrt(torch.pow(a - b, 2).sum(2))

    def open_classify(self, data, features, gb_centroids, gb_radii, gb_labels):
        # 计算输入特征与所有质心之间的欧氏距离
        gb_centroids=gb_centroids.to(self.device)
        gb_radii=gb_radii.to(self.device)
        gb_labels=gb_labels.to(self.device)
        features=features.to(self.device)
        logits = self.euclidean_metric(features, gb_centroids)
        _, preds = logits.min(dim=1)



        euc_dis = torch.norm(features - gb_centroids[preds], dim=1)


        final_preds = torch.tensor([data.unseen_token_id] * features.shape[0], device=features.device)


        for i in range(features.shape[0]):
            if euc_dis[i] < gb_radii[preds[i]]:
                final_preds[i] = gb_labels[preds[i]]
            else:
                final_preds[i] = data.unseen_token_id
        return final_preds




    def evaluation(self, args, data, gb_centroids, gb_radii, gb_labels,mode="eval"):
        self.model.eval()
        print(gb_labels.shape)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)
        if mode == 'eval':
            dataloader = data.eval_dataloader
        elif mode == 'test':
            dataloader = data.test_dataloader

        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids,orignal_label, label_ids,index= batch
            with torch.set_grad_enabled(False):
                pooled_output, _ = self.model(input_ids, segment_ids,
                                              input_mask)
                preds = self.open_classify(data,pooled_output,gb_centroids, gb_radii, gb_labels)

                total_labels = torch.cat((total_labels, label_ids))
                total_preds = torch.cat((total_preds, preds))

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        self.predictions = list([data.label_list[idx] for idx in y_pred])
        self.true_labels = list([data.label_list[idx] for idx in y_true])

        if mode == 'eval':
            cm = confusion_matrix(y_true, y_pred)
            eval_score = F_measure(cm)['F1-score']
            return eval_score

        elif mode == 'test':

            cm = confusion_matrix(y_true, y_pred)
            results = F_measure(cm)
            acc = round(accuracy_score(y_true, y_pred) * 100, 2)
            results['Accuracy'] = acc

            self.test_results = results
            self.save_results(args)
            print('Accuracy:', acc)






    def restore_model(self, args):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        self.model.load_state_dict(torch.load(output_model_file))

    

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [
            args.dataset,
            args.known_cls_ratio,
            args.ind_noise_ratio,
            args.ood_noise_ratio,
            args.purity_select_ball,
            args.min_ball_select_ball,
            args.warm_train_epoch,
            args.ood_type,
            args.train_batch_size,
            args.eval_batch_size,
            args.num_train_epochs,
            args.wait_patient,
            args.lr,
            args.alpha,
            args.beta,
            args.gamma,
            args.GIND_p,
            args.SOOD_p,
            args.seed
        ]


        names = [
            'dataset',
            'known_cls_ratio',
            'ind_noise_ratio',
            'ood_noise_ratio',
            'purity_select_ball',
            'min_ball_select_ball',
            'warm_train_epoch',
            'ood_type',
            'train_batch_size',
            'eval_batch_size',
            'num_train_epochs',
            'wait_patient',
            'lr',
            'alpha',
            'beta',
            'gamma',
            'GIND_p',
            'SOOD_p',
            'seed'
        ]



        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(self.test_results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values())
        file_name = 'results.csv'
        results_path = os.path.join(args.save_results_path, file_name)
        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori,columns = keys)
            df1.to_csv(results_path,index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results,index=[1])

            df1 = pd.concat([df1, new], ignore_index=True)
            df1.to_csv(results_path,index=False)
        data_diagram = pd.read_csv(results_path)
        
        print('test_results', data_diagram)


    
    
