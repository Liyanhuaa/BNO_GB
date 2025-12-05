This is the source code for "Beneficial Noise Learning for Open Intent Classification via Granular-Ball Representation"

We thank the code of "MGNR: A Multi-Granularity Neighbor Relationship  and Its Application in KNN Classification and  Clustering Methods" from "https://github.com/xjnine/MGNR"


We thank the code of "Deep Open Intent Classification with Adaptive Decision Boundary" from "https://github.com/thuiar/Adaptive-Decision-Boundary"


Run example:
--dataset
stackoverflow
--open_noise_dataset
snips
--known_cls_ratio
0.25
--ind_noise_ratio
0.1
--ood_noise_ratio
0.1
--seed
0
--freeze_bert_parameters
--gpu_id
0
--train_batch_size
128
--eval_batch_size
1024
--num_train_epochs
100