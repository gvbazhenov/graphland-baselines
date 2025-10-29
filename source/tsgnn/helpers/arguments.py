from argparse import ArgumentParser

from helpers.gnn_type import GNNType
from helpers.train_test_setup import TrainTestSetup


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--project", dest="project", type=str, required=True)

    # data
    parser.add_argument("--is_train", dest="is_train", default=False, action='store_true', required=False)
    parser.add_argument("--train_test_setup", dest="train_test_setup", default=TrainTestSetup.trainset1,
                        type=TrainTestSetup.from_string, choices=list(TrainTestSetup), required=False)
    parser.add_argument("--train_size", dest="train_size", default=1, type=int, required=False)

    # GFM
    parser.add_argument("--gnn_type", dest="gnn_type", default=GNNType.MEAN_GNN, type=GNNType.from_string,
                        choices=list(GNNType), required=False)
    parser.add_argument("--hid_channel", dest="hid_channel", default=16, type=int, required=False)
    parser.add_argument("--num_layers", dest="num_layers", default=2, type=int, required=False)
    parser.add_argument("--ls_num_layers", dest="ls_num_layers", default=-1, type=int, required=False)

    # optimization
    parser.add_argument("--lp_ratio", dest="lp_ratio", default=0.5, type=float, required=False)
    parser.add_argument("--max_epochs", dest="max_epochs", default=500, type=int, required=False)
    parser.add_argument(
        "--checkpoints",
        dest="checkpoints",
        type=int,
        nargs='+',  # or '*' if empty list is acceptable
        default=[100, 300, 500, 1000, 2000],
        required=False,
    )
    parser.add_argument("--lr", dest="lr", default=1e-3, type=float, required=False)

    # gpu
    parser.add_argument('--gpu', dest="gpu", type=int, required=False)
    return parser.parse_args()
