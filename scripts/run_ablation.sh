python main.py --dataset arxiv-year --sub_dataset None --method mlpnorm_improve --epochs 100 --hidden_channels 256 \
 --lr 0.005 --dropout 0.7 --weight_decay 0.001 --alpha 0.0 --beta 1.0 --gamma 0.7 --delta 0.1 \
 --norm_func_id 1 --norm_layers 3 --orders_func_id 2 --z1 0.1 --z2 0.0 \
 --orders 3 --display_step 1 --runs 5