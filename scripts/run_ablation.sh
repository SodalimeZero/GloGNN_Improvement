python train.py --dataset roman-empire --model mlpnorm_improve --num_layers 2 --hidden_dim 256 --use_sgc_features --device cuda:0\
 --num_steps 500 --alpha 0.0 --beta 0.4 --gamma 0.8 --delta 0.4 --dropout 0.9 --num_runs 10 \
  --lr 3e-5 --norm_func_id 2 --norm_layers 1 --orders_func_id 1 --orders 2 --z1 0.4 --z2 0.1 --without_initial

python train.py --dataset amazon-ratings --model mlpnorm_improve --num_layers 2 --hidden_dim 256 --use_sgc_features --device cuda:0\
 --num_steps 500 --alpha 0.0 --beta 0.4 --gamma 0.8 --delta 0.4 --dropout 0.9 --num_runs 10 \
  --lr 3e-5 --norm_func_id 2 --norm_layers 1 --orders_func_id 1 --orders 2 --z1 0.3 --z2 0.2 --without_initial

python train.py --dataset tolokers --model mlpnorm_improve --num_layers 2 --hidden_dim 256 --use_sgc_features --device cuda:0\
 --num_steps 500 --alpha 0.0 --beta 0.4 --gamma 0.8 --delta 0.4 --dropout 0.9 --num_runs 10 \
  --lr 3e-5 --norm_func_id 2 --norm_layers 1 --orders_func_id 1 --orders 2 --z1 0.4 --z2 0.5 --without_initial

python train.py --dataset minesweeper --model mlpnorm_improve --num_layers 2 --hidden_dim 256 --use_sgc_features --device cuda:0\
 --num_steps 500 --alpha 0.0 --beta 0.4 --gamma 0.8 --delta 0.4 --dropout 0.9 --num_runs 10 \
  --lr 3e-5 --norm_func_id 2 --norm_layers 1 --orders_func_id 1 --orders 2 --z1 0.7 --z2 0.1 --without_initial

python train.py --dataset questions --model mlpnorm_improve --num_layers 2 --hidden_dim 256 --use_sgc_features --device cuda:0\
 --num_steps 500 --alpha 0.0 --beta 0.4 --gamma 0.8 --delta 1.0 --dropout 0.2 --num_runs 10 \
  --lr 3e-5 --norm_func_id 2 --norm_layers 1 --orders_func_id 1 --orders 2 --z1 0.4 --z2 0.3 --without_initial

python train.py --dataset roman-empire --model mlpnorm_improve --num_layers 2 --hidden_dim 256 --use_sgc_features --device cuda:0\
 --num_steps 500 --alpha 0.0 --beta 0.4 --gamma 0.8 --delta 0.4 --dropout 0.9 --num_runs 10 \
  --lr 3e-5 --norm_func_id 2 --norm_layers 1 --orders_func_id 1 --orders 2 --z1 0.4 --z2 0.1 --without_topology

python train.py --dataset amazon-ratings --model mlpnorm_improve --num_layers 2 --hidden_dim 256 --use_sgc_features --device cuda:0\
 --num_steps 500 --alpha 0.0 --beta 0.4 --gamma 0.8 --delta 0.4 --dropout 0.9 --num_runs 10 \
  --lr 3e-5 --norm_func_id 2 --norm_layers 1 --orders_func_id 1 --orders 2 --z1 0.3 --z2 0.2 --without_topology

python train.py --dataset tolokers --model mlpnorm_improve --num_layers 2 --hidden_dim 256 --use_sgc_features --device cuda:0\
 --num_steps 500 --alpha 0.0 --beta 0.4 --gamma 0.8 --delta 0.4 --dropout 0.9 --num_runs 10 \
  --lr 3e-5 --norm_func_id 2 --norm_layers 1 --orders_func_id 1 --orders 2 --z1 0.4 --z2 0.5 --without_topology

python train.py --dataset minesweeper --model mlpnorm_improve --num_layers 2 --hidden_dim 256 --use_sgc_features --device cuda:0\
 --num_steps 500 --alpha 0.0 --beta 0.4 --gamma 0.8 --delta 0.4 --dropout 0.9 --num_runs 10 \
  --lr 3e-5 --norm_func_id 2 --norm_layers 1 --orders_func_id 1 --orders 2 --z1 0.7 --z2 0.1 --without_topology

python train.py --dataset questions --model mlpnorm_improve --num_layers 2 --hidden_dim 256 --use_sgc_features --device cuda:0\
 --num_steps 500 --alpha 0.0 --beta 0.4 --gamma 0.8 --delta 1.0 --dropout 0.2 --num_runs 10 \
  --lr 3e-5 --norm_func_id 2 --norm_layers 1 --orders_func_id 1 --orders 2 --z1 0.4 --z2 0.3 --without_topology