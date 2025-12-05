from init_parameter import *
from dataloader import *
from pretrain import *
from torch.utils.tensorboard import SummaryWriter
# from tensorflow.keras.callbacks import TensorBoard
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from utils import util
import time
from gb_test import ModelManager




if __name__ == '__main__':
    start_time = time.time()
    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    print('Parameters Initialization...')

    parser = init_model()
    args = parser.parse_args()
    data = Data(args)

    print('Training begin...')
    manager_p1 = PretrainModelManager(args, data)
    best_original_examples=manager_p1.train(args, data)
    print('Training finished!')



    if args.dataset=="stackoverflow":
        if args.ind_noise_ratio==0.1:
            min_ball_list=[12]
            purity_list = [0.9]
        if args.ind_noise_ratio==0.2:
            min_ball_list=[8]
            purity_list = [0.8]
    if args.dataset=="snips":
        min_ball_list=[80]
        purity_list = [0.9]
    if args.dataset=="banking":
        min_ball_list=[4,6,8,10]
        purity_list = [0.5, 0.6, 0.7, 0.8, 0.9]



    for purity in purity_list:
        for min_ball in min_ball_list:
            print(f"\n=== Start evaluation with purity={purity}, min_ball={min_ball} ===")

            args.purity_select_ball = purity
            args.min_ball_select_ball = min_ball


            start_time = time.time()

            print('Calculate ball begin...')
            gb_centroids, gb_radii, gb_labels = manager_p1.calculate_granular_balls(args, data)
            if len(gb_centroids) == 0:
                continue
            else:
                print('Calculate ball finished!')
                print('len_gb',len(gb_centroids))

                manager = ModelManager(args, data, manager_p1.model)

                print('Evaluation begin...')
                manager.evaluation(args, data, gb_centroids, gb_radii, gb_labels, mode="test")
                print('Evaluation finished!')

                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Elapsed time for purity={purity}, min_ball={min_ball}: {elapsed_time:.2f} seconds")


                del gb_centroids, gb_radii, gb_labels
                import gc

                gc.collect()
                import torch

                torch.cuda.empty_cache()