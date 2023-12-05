import argparse
import numpy as np
from torch_sparse import SparseTensor
from tqdm import tqdm

import torch
from torch.cuda.amp import autocast, GradScaler
from models.GloGNN import MLPNORM
from models.GloGNN_Improve import MLPNORM_IMPROVE
from dataset.dataset import Dataset
from utils.utils import Logger, get_parameter_groups, get_lr_scheduler_with_warmup
import scipy.sparse as sp
from sklearn.preprocessing import normalize as sk_normalize


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default=None, help='Experiment name. If None, model name is used.')
    parser.add_argument('--save_dir', type=str, default='results', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default='roman-empire',
                        choices=['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions',
                                 'squirrel', 'squirrel-directed', 'squirrel-filtered', 'squirrel-filtered-directed',
                                 'chameleon', 'chameleon-directed', 'chameleon-filtered', 'chameleon-filtered-directed',
                                 'actor', 'texas', 'texas-4-classes', 'cornell', 'wisconsin'])

    # model architecture
    parser.add_argument('--model', type=str, default='GT-sep',
                        choices=['GloGNN', 'mlpnorm_improve'])
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--hidden_dim_multiplier', type=float, default=1)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--normalization', type=str, default='LayerNorm', choices=['None', 'LayerNorm', 'BatchNorm'])

    # regularization
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0)

    # training parameters
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--num_warmup_steps', type=int, default=None,
                        help='If None, warmup_proportion is used instead.')
    parser.add_argument('--warmup_proportion', type=float, default=0, help='Only used if num_warmup_steps is None.')

    # node feature augmentation
    parser.add_argument('--use_sgc_features', default=False, action='store_true')
    parser.add_argument('--use_identity_features', default=False, action='store_true')
    parser.add_argument('--use_adjacency_features', default=False, action='store_true')
    parser.add_argument('--do_not_use_original_features', default=False, action='store_true')

    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')

    # glognn
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='Weight for frobenius norm on Z.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for frobenius norm on Z-A')
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='Weight for MLP results kept')
    parser.add_argument('--delta', type=float, default=0.0,
                        help='Weight for node features, thus 1-delta for adj')
    parser.add_argument('--orders_func_id', type=int, default=2, choices=[1, 2, 3],
                        help='Sum function of adj orders in norm layer, ids \in [1, 2, 3]')
    parser.add_argument('--orders', type=int, default=1,
                        help='Number of adj orders in norm layer')
    parser.add_argument('--norm_func_id', type=int, default=2, choices=[1, 2],
                        help='Function of norm layer, ids \in [1, 2]')
    parser.add_argument('--norm_layers', type=int, default=1,
                        help='Number of groupnorm layers')
    parser.add_argument('--z1', type=float, default=1.0,
                        help='Weight for frobenius norm on Z-A')
    parser.add_argument('--z2', type=float, default=1.0,
                        help='Weight for frobenius norm on Z-A')
    parser.add_argument('--without_initial', default=False, action='store_true')
    parser.add_argument('--without_topology', default=False, action='store_true')

    args = parser.parse_args()

    if args.name is None:
        args.name = args.model

    return args

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def row_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_normalized = sk_normalize(adj, norm='l1', axis=1)
    # row_sum = np.array(adj.sum(1))
    # row_sum = (row_sum == 0)*1+row_sum
    # adj_normalized = adj/row_sum
    return sp.coo_matrix(adj_normalized)

def train_step(model, dataset, optimizer, scheduler, scaler, amp=False):
    model.train()

    with autocast(enabled=amp):
        args = get_args()
        if args.model == 'GloGNN' or args.model == 'mlpnorm_improve' or args.model == 'mlpnorm_improve1':
            edge_index = torch.stack(dataset.graph.adj_tensors(fmt='coo'), dim=0)
            adj = SparseTensor(row=edge_index[0].long(), col=edge_index[1].long(), sparse_sizes=(
                    dataset.graph.number_of_nodes(), dataset.graph.number_of_nodes())
                ).to_torch_sparse_coo_tensor()
            # adj = adj.to_dense()
            logits = model(x=dataset.node_features, adj=adj.to(args.device))
        else:
            logits = model(graph=dataset.graph, x=dataset.node_features)
        loss = dataset.loss_fn(input=torch.squeeze(logits[dataset.train_idx]), target=dataset.labels[dataset.train_idx])

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()


@torch.no_grad()
def evaluate(model, dataset, amp=False):
    model.eval()

    with autocast(enabled=amp):
        args = get_args()
        if args.model == 'GloGNN' or args.model == 'mlpnorm_improve' or args.model == 'mlpnorm_improve1':
            edge_index = torch.stack(dataset.graph.adj_tensors(fmt='coo'), dim=0)
            adj = SparseTensor(row=edge_index[0].long(), col=edge_index[1].long(), sparse_sizes=(
                    dataset.graph.number_of_nodes(), dataset.graph.number_of_nodes())
                ).to_torch_sparse_coo_tensor()
            # adj = adj.to_dense()
            logits = model(x=dataset.node_features, adj=adj.to(args.device))
        else:
            logits = model(graph=dataset.graph, x=dataset.node_features)
    metrics = dataset.compute_metrics(logits)

    return metrics


def main():
    args = get_args()

    torch.manual_seed(0)

    dataset = Dataset(name=args.dataset,
                      add_self_loops=False,
                      device=args.device,
                      use_sgc_features=args.use_sgc_features,
                      use_identity_features=args.use_identity_features,
                      use_adjacency_features=args.use_adjacency_features,
                      do_not_use_original_features=args.do_not_use_original_features)

    logger = Logger(args, metric=dataset.metric, num_data_splits=dataset.num_data_splits)

    for run in range(1, args.num_runs + 1):
        if args.model == 'mlpnorm_improve':
            model = MLPNORM_IMPROVE(nnodes=dataset.node_features.shape[0], 
                            nfeat=dataset.num_node_features, 
                            nhid=args.hidden_dim, 
                            nclass=dataset.num_targets, 
                            dropout=args.dropout, 
                            alpha=args.alpha, 
                            beta=args.beta, 
                            gamma=args.gamma, 
                            delta=args.delta,
                            norm_func_id=args.norm_func_id, 
                            norm_layers=args.norm_layers, 
                            orders=args.orders, 
                            orders_func_id=args.orders_func_id, 
                            device=args.device,
                            out_channels=dataset.num_targets, 
                            num_layers=args.num_layers,
                            z1=args.z1,
                            z2=args.z2,
                            without_initial=args.without_initial,
                            without_topology=args.without_topology)
        elif args.model == 'GloGNN':
            model = MLPNORM(nnodes=dataset.node_features.shape[0], 
                            nfeat=dataset.num_node_features, 
                            nhid=args.hidden_dim, 
                            nclass=dataset.num_targets, 
                            dropout=args.dropout, 
                            alpha=args.alpha, 
                            beta=args.beta, 
                            gamma=args.gamma, 
                            delta=args.delta,
                            norm_func_id=args.norm_func_id, 
                            norm_layers=args.norm_layers, 
                            orders=args.orders, 
                            orders_func_id=args.orders_func_id, 
                            device=args.device)

        model.to(args.device)

        parameter_groups = get_parameter_groups(model)
        optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
        scaler = GradScaler(enabled=args.amp)
        scheduler = get_lr_scheduler_with_warmup(optimizer=optimizer, num_warmup_steps=args.num_warmup_steps,
                                                 num_steps=args.num_steps, warmup_proportion=args.warmup_proportion)

        logger.start_run(run=run, data_split=dataset.cur_data_split + 1)
        with tqdm(total=args.num_steps, desc=f'Run {run}', disable=args.verbose) as progress_bar:
            for step in range(1, args.num_steps + 1):
                train_step(model=model, dataset=dataset, optimizer=optimizer, scheduler=scheduler,
                           scaler=scaler, amp=args.amp)
                metrics = evaluate(model=model, dataset=dataset, amp=args.amp)
                logger.update_metrics(metrics=metrics, step=step)

                progress_bar.update()
                progress_bar.set_postfix({metric: f'{value:.2f}' for metric, value in metrics.items()})

        logger.finish_run()
        model.cpu()
        dataset.next_data_split()

    logger.print_metrics_summary()


if __name__ == '__main__':
    main()
