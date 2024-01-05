python main.py --dataset arxiv-year --sub_dataset None --method mlpnorm_improve --epochs 100 --hidden_channels 256 \
 --lr 0.005 --dropout 0.7 --weight_decay 0.001 --alpha 0.0 --beta 1.0 --gamma 0.7 --delta 0.1 \
 --norm_func_id 1 --norm_layers 3 --orders_func_id 2 --z1 1.0 --z2 0.2 \
 --orders 3 --display_step 1 --runs 5
python main.py --dataset genius --sub_dataset None --method mlpnorm_improve --epochs 100 --hidden_channels 256 \
 --lr 0.001 --dropout 0.0 --weight_decay 0.0 --alpha 0.0 --beta 1.0 --gamma 0.9 --delta 0.9 --z1 0.8 --z2 0.2\
 --norm_func_id 1 --norm_layers 2 --orders_func_id 2 --orders 2 --display_step 1 --runs 5 

python main.py --dataset pokec --sub_dataset None --method mlpnorm_improve --epochs 500 --hidden_channels 128\
 --lr 0.005 --dropout 0.8 --weight_decay 0.01 --alpha 0.0 --beta 1.0 --gamma 0.5 --delta 0.5 --z1 1.0 --z2 0.2 \
 --norm_func_id 1 --norm_layers 1 --orders_func_id 2 --orders 3 --display_step 1 --runs 5 --directed

python main.py --dataset fb100 --sub_dataset Penn94 --method mlpnorm_improve --epochs 500 --hidden_channels 256\
 --lr 0.001 --dropout 0.8 --weight_decay 0.1 --alpha 0.0 --beta 1.0 --gamma 0.6 --delta 0.5 --z1 0.5 --z2 0.5 \
 --norm_func_id 1 --norm_layers 1 --orders_func_id 2 --orders 3 --display_step 1 --runs 5 --directed

python main.py --dataset twitch-gamer --sub_dataset None --method mlpnorm_improve --epochs 500 --hidden_channels 256\
 --lr 0.01 --dropout 0.8 --weight_decay 0.01 --alpha 0.0 --beta 1.0 --gamma 0.1 --delta 1.0 --z1 0.7 --z2 0.2 \
 --norm_func_id 1 --norm_layers 1 --orders_func_id 2 --orders 3 --display_step 1 --runs 5 --directed

 python main.py --dataset snap-patents --sub_dataset None --method mlpnorm_improve --epochs 500 --hidden_channels 64\
 --lr 0.005 --dropout 0.8 --weight_decay 0.01 --alpha 0.0 --beta 1.0 --gamma 0.6 --delta 0.5 --z1 1.0 --z2 0.3 \
 --norm_func_id 1 --norm_layers 1 --orders_func_id 2 --orders 3 --display_step 1 --runs 5 --directed

python train.py --dataset roman-empire --model mlpnorm_improve --num_layers 2 --hidden_dim 256 --use_sgc_features --device cuda:0\
 --num_steps 500 --alpha 0.0 --beta 0.4 --gamma 0.8 --delta 0.4 --dropout 0.9 --num_runs 10 \
  --lr 3e-5 --norm_func_id 2 --norm_layers 1 --orders_func_id 1 --orders 2 --z1 0.4 --z2 0.1

python train.py --dataset amazon-ratings --model mlpnorm_improve --num_layers 2 --hidden_dim 256 --use_sgc_features --device cuda:0\
 --num_steps 500 --alpha 0.0 --beta 0.4 --gamma 0.8 --delta 0.4 --dropout 0.9 --num_runs 10 \
  --lr 3e-5 --norm_func_id 2 --norm_layers 1 --orders_func_id 1 --orders 2 --z1 0.3 --z2 0.2

python train.py --dataset tolokers --model mlpnorm_improve --num_layers 2 --hidden_dim 256 --use_sgc_features --device cuda:0\
 --num_steps 500 --alpha 0.0 --beta 0.4 --gamma 0.8 --delta 0.4 --dropout 0.9 --num_runs 10 \
  --lr 3e-5 --norm_func_id 2 --norm_layers 1 --orders_func_id 1 --orders 2 --z1 0.4 --z2 0.5

python train.py --dataset minesweeper --model mlpnorm_improve --num_layers 2 --hidden_dim 256 --use_sgc_features --device cuda:0\
 --num_steps 500 --alpha 0.0 --beta 0.4 --gamma 0.8 --delta 0.4 --dropout 0.9 --num_runs 10 \
  --lr 3e-5 --norm_func_id 2 --norm_layers 1 --orders_func_id 1 --orders 2 --z1 0.7 --z2 0.1

python train.py --dataset questions --model mlpnorm_improve --num_layers 2 --hidden_dim 256 --use_sgc_features --device cuda:0\
 --num_steps 500 --alpha 0.0 --beta 0.4 --gamma 0.8 --delta 1.0 --dropout 0.2 --num_runs 10 \
  --lr 3e-5 --norm_func_id 2 --norm_layers 1 --orders_func_id 1 --orders 2 --z1 0.4 --z2 0.3
