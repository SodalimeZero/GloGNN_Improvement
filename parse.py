from numpy import select
from models.GloGNN import MLPNORM
from models.GloGNN_Improve import MLPNORM_IMPROVE
from utils.data_utils import normalize
import math


def parse_method(args, dataset, n, c, d, device, num_relations):
    if args.dataset in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        if args.method == 'mlpnorm_improve':
            model = MLPNORM_IMPROVE(nnodes=dataset.node_features.shape[0], nfeat=dataset.num_node_features, nhid=args.hidden_channels, nclass=dataset.num_targets, dropout=args.dropout, alpha=args.alpha, 
                            beta=args.beta, gamma=args.gamma, delta=args.delta, norm_func_id=args.norm_func_id, 
                            norm_layers=args.norm_layers, orders=args.orders, orders_func_id=args.orders_func_id, 
                            device=device, out_channels=dataset.num_targets, num_layers=args.num_layers, z1=args.z1, z2=args.z2)
        elif args.method == 'GloGNN':
            model = MLPNORM(nnodes=dataset.node_features.shape[0], 
                            nfeat=dataset.num_node_features, 
                            nhid=args.hidden_channels, 
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
                            device=device) 
    else:
        if args.method == 'mlpnorm':
            model = MLPNORM(nnodes=dataset.graph['num_nodes'], nfeat=d, nhid=args.hidden_channels, nclass=c, dropout=args.dropout, alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                            delta=args.delta, norm_func_id=args.norm_func_id, norm_layers=args.norm_layers, orders_func_id=args.orders_func_id, orders=args.orders, device=device).to(device)
        elif args.method == 'mlpnorm_improve':
            model = MLPNORM_IMPROVE(nnodes=dataset.graph['num_nodes'], nfeat=d, nhid=args.hidden_channels, nclass=c, dropout=args.dropout, alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                            delta=args.delta, norm_func_id=args.norm_func_id, norm_layers=args.norm_layers, orders_func_id=args.orders_func_id, 
                            orders=args.orders, device=device, out_channels=c, num_layers=args.num_layers, z1=args.z1, z2=args.z2).to(device)
        else:
            raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    parser.add_argument('--dataset', type=str, default='ogbn-proteins')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--method', '-m', type=str, default='mlpnorm')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rocauc', action='store_true',
                        help='set the eval function to rocauc')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--sampling', action='store_true',
                        help='use neighbor sampling')
    parser.add_argument('--inner_activation', action='store_true',
                        help='Whether linkV3 uses inner activation')
    parser.add_argument('--inner_dropout', action='store_true',
                        help='Whether linkV3 uses inner dropout')
    parser.add_argument("--SGD", action='store_true',
                        help='Use SGD as optimizer')
    parser.add_argument('--link_init_layers_A', type=int, default=1)
    parser.add_argument('--link_init_layers_X', type=int, default=1)
    parser.add_argument('--adam', action='store_true',
                        help='use adam instead of adamW')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    
    parser.add_argument('--use_sgc_features', default=False, action='store_true')
    parser.add_argument('--use_identity_features', default=False, action='store_true')
    parser.add_argument('--use_adjacency_features', default=False, action='store_true')
    parser.add_argument('--do_not_use_original_features', default=False, action='store_true')

    # used for mlpnorm
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='Weight for frobenius norm on Z.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for frobenius norm on Z-A')
    parser.add_argument('--beta1', type=float, default=1.0,
                        help='Weight for frobenius norm on Z-A')
    parser.add_argument('--z1', type=float, default=1.0,
                        help='Weight for frobenius norm on Z-A')
    parser.add_argument('--z2', type=float, default=1.0,
                        help='Weight for frobenius norm on Z-A')
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='Weight for MLP results kept')
    parser.add_argument('--delta', type=float, default=0.0,
                        help='Weight for node features, thus 1-delta for adj')
    parser.add_argument('--norm_func_id', type=int, default=2,
                        help='Function of norm layer, ids \in [1, 2]')
    parser.add_argument('--norm_layers', type=int, default=1,
                        help='Number of groupnorm layers')
    parser.add_argument('--orders_func_id', type=int, default=2,
                        help='Sum function of adj orders in norm layer, ids \in [1, 2, 3]')
    parser.add_argument('--orders', type=int, default=1,
                        help='Number of adj orders in norm layer')
