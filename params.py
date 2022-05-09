import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')

    # for model
    parser.add_argument('--target_behavior', default='buy', type=str)
    parser.add_argument('--hidden_dim', default=16, type=int)
    parser.add_argument('--layer_num', default=3, type=int)
    parser.add_argument('--drop_rate', default=0.5, type=float)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--gnn_opt_base_lr', default=1e-4, type=float)
    parser.add_argument('--gnn_opt_max_lr', default=5e-4, type=float)
    parser.add_argument('--gnn_opt_weight_decay', default=1e-4, type=float)
    parser.add_argument('--meta_lr', default=1e-3, type=float)
    parser.add_argument('--meta_opt_base_lr', default=1e-4, type=float)
    parser.add_argument('--meta_opt_max_lr', default=5e-4, type=float)
    parser.add_argument('--meta_opt_weight_decay', default=1e-3, type=float)
    parser.add_argument('--CL_mini_batch_size', default=15, type=int)
    parser.add_argument('--reg', default=1e-2, type=float)
    parser.add_argument('--beta', default=1e-3, type=float)

    # for test
    parser.add_argument('--shoot', default=10, type=int)

    # for save and read
    parser.add_argument('--dataset_path', default='D:/hku_intern/dataset/IJCAI_15/', type=str)

    # for train
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--meta_batch', default=128, type=int)
    parser.add_argument('--epoch_num', default=200, type=int)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--samp_num', default=40, type=int)

    return parser.parse_args()


args = parse_args()
