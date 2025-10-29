import copy
from argparse import Namespace
import torch
import sys
import tqdm
from typing import Tuple, Any, List
import neptune
from neptune.utils import stringify_unsupported
from torch_geometric.data import Data, DataLoader
import os.path as osp
import os
import yaml
from torch import Tensor
from enum import Enum

from helpers.constants import ROOT_DIR, SEEDS, TASK_LOSS
from helpers.utils import set_seed, coo_to_csr, str_print, accuracy
from models.gfm import GFMArgs, GFM
from helpers.datasets import DataSet
from helpers.constants import API_TOKEN
from helpers.metrics import LossAndMetric
from helpers.split_data import split_data_per_fold

from sklearn.metrics import average_precision_score, accuracy_score

RESULTS_ROOT = './results'
os.makedirs(RESULTS_ROOT, exist_ok=True)


def save_results(results: dict, name: str) -> None:
    name = name.removeprefix('graphland_').replace('_', '-')
    path = f'{RESULTS_ROOT}/{name}.yaml'
    with open(path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)


def create_model_dict_path(args: Namespace) -> str:
    model_params = []
    for arg in vars(args):
        if arg not in ['project', 'is_train', 'gpu', 'checkpoints']:
            value = getattr(args, arg)
            if value is not None:
                if isinstance(value, Enum):
                    model_params.append(value.name)
                elif isinstance(value, bool):
                    if value:
                        model_params.append(arg)
                else:
                    model_params.append(str(value))
    return osp.join(ROOT_DIR, 'saved_models', '_'.join(model_params))


class Experiment(object):
    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args
        self.save_load_path = create_model_dict_path(args=args)
        # self.neptune_logger = neptune.init_run(project=args.project, api_token='API_TOKEN')  # your credentials
        for arg in vars(args):
            value_arg = getattr(args, arg)
            print(f"{arg}: {value_arg}")
            self.__setattr__(arg, value_arg)

        # self.neptune_logger["params"] = stringify_unsupported({arg: getattr(args, arg) for arg in vars(args)})
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_single_data(self):
        mean_lists, std_lists = [], []
        for eval_dataset in self.train_test_setup.get_test_datasets():
            try:
                dataset_mean, dataset_std = self.run_multiple_data(is_train=False, dataset_list=[eval_dataset])
            except torch.cuda.OutOfMemoryError:
                dataset_mean = torch.tensor([torch.nan, torch.nan])
                dataset_std = torch.tensor([torch.nan, torch.nan])
            torch.cuda.empty_cache()
            # print(dataset_mean, dataset_std)
            results = {
                'mean': {
                    'val': dataset_mean[0].item(),
                    'test': dataset_mean[1].item(),
                },
                'std': {
                    'val': dataset_std[0].item(),
                    'test': dataset_std[1].item(),
                }
            }
            save_results(results, eval_dataset.name)
            # TODO: save metrics mean & std
            mean_lists.append(dataset_mean)
            std_lists.append(dataset_std)
        metric_mean = torch.stack(mean_lists, dim=0).mean(dim=0)
        std_mean = torch.stack(std_lists, dim=0).mean(dim=0)

        # record
        print(f'FINAL ' + str_print(metric_mean=metric_mean, metric_std=std_mean))
        for idx, name in enumerate(['val', 'test']):
            mean_log = str_print(suffix=f"{name}_metric_mean")
            std_log = str_print(suffix=f"{name}_metric_std")
            # self.neptune_logger[mean_log] = metric_mean[idx].item()
            # self.neptune_logger[std_log] = std_mean[idx].item()

    def run_multiple_data(self, is_train: bool, dataset_list: List[DataSet]) -> Tuple[Tensor, Tensor]:
        # load model args
        gfm_args = GFMArgs(gnn_type=self.gnn_type, hid_channel=self.hid_channel, num_layers=self.num_layers,
                           ls_num_layers=self.ls_num_layers, lp_ratio=self.lp_ratio)

        # seeds
        metrics_list = []
        for seed in SEEDS:
            data_list = []
            for dataset in dataset_list:
                set_seed(seed=seed)
                data = dataset.load()
                data = split_data_per_fold(seed=seed, data=data, ls_num_layers=self.ls_num_layers,
                                           dataset_name=dataset.name)
                setattr(data, 'obj', dataset)
                data_list.append(data)
            set_seed(seed=seed)
            best_losses_n_metric = self.single_seed(is_train=is_train, data_list=data_list, gfm_args=gfm_args,
                                                    seed=seed)
            metrics_list.append(best_losses_n_metric.get_fold_metrics())

        metrics_mean = torch.stack(metrics_list, dim=0).mean(dim=0)  # (2,)
        metrics_std = torch.stack(metrics_list, dim=0).std(dim=0)  # (2,)

        # record
        if is_train:
            print(str_print(train_test_name=self.train_test_setup.name,
                            metric_mean=metrics_mean, metric_std=metrics_std) + '\n')
        else:
            print(str_print(train_test_name=self.train_test_setup.name, single_dataset_name=dataset_list[0].name,
                            metric_mean=metrics_mean, metric_std=metrics_std) + '\n')
        for idx, name in enumerate(['val', 'test']):
            if is_train:
                mean_log = str_print(suffix=f"{name}_metric_mean")
                std_log = str_print(suffix=f"{name}_metric_std")
            else:
                mean_log = str_print(single_dataset_name=dataset_list[0].name,
                                     suffix=f"{name}_metric_mean")
                std_log = str_print(single_dataset_name=dataset_list[0].name,
                                    suffix=f"{name}_metric_std")
            # self.neptune_logger[mean_log] = metrics_mean[idx].item()
            # self.neptune_logger[std_log] = metrics_std[idx].item()
        return metrics_mean, metrics_std

    def single_seed(self, is_train: bool, data_list: List[Data], gfm_args: GFMArgs, seed: int) -> LossAndMetric:
        # convert to triton representation
        for data in data_list:
            if gfm_args.gnn_type.uses_triton():
                rowptr, indices = coo_to_csr(data.edge_index[0], data.edge_index[1], num_nodes=data.x.shape[0])
                setattr(data, 'edge_index', [])
                setattr(data, 'rowptr', rowptr)
                setattr(data, 'indices', indices)
            else:
                setattr(data, 'rowptr', [])
                setattr(data, 'indices', [])

        model = GFM(gfm_args=gfm_args)
        model_path = osp.join(self.save_load_path, f"Seed{seed}.pt")
        if not is_train or os.path.exists(model_path):
            if is_train:
                print('A trained model is already saved! Loading....')
            # load
            model_load_path = torch.load(model_path, weights_only=True)
            model.load_state_dict(model_load_path)
            model = model.to(device=self.device)

            # test
            loader = DataLoader(data_list, batch_size=1, shuffle=False)
            val_loss, val_metric = self.test(loader=loader, model=model, mask_str='val')
            test_loss, test_metric = self.test(loader=loader, model=model, mask_str='test')

            # record
            losses_n_metric = \
                LossAndMetric(val_loss=val_loss, test_loss=test_loss,
                              val_metric=val_metric, test_metric=test_metric)
            print(f'seedFINAL ' + str_print(train_test_name=self.train_test_setup.name,
                                            single_dataset_name=data_list[0].obj.name,
                                            seed=seed, losses_n_metric=losses_n_metric))
            for name in losses_n_metric._fields:
                losses_n_metric_by_split = getattr(losses_n_metric, name)
                log = str_print(single_dataset_name=data_list[0].obj.name, seed=seed,
                                suffix=f"{name}")
                # self.neptune_logger[log] = losses_n_metric_by_split
        else:
            # train
            model = model.to(device=self.device)
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.lr)
            with tqdm.tqdm(total=self.max_epochs, file=sys.stdout) as pbar:
                losses_n_metric, state_dict =\
                    self.trainer(data_list=data_list, model=model, seed=seed, optimizer=optimizer, pbar=pbar)

            # save
            os.makedirs(self.save_load_path, exist_ok=True)
            with open(model_path, "wb") as f:
                torch.save(state_dict, f)

            # record
            print(f'seedFINAL ' + str_print(train_test_name=self.train_test_setup.name, seed=seed,
                                            losses_n_metric=losses_n_metric))
            for name in losses_n_metric._fields:
                losses_n_metric_by_split = getattr(losses_n_metric, name)
                log = str_print(seed=seed,
                                suffix=f"{name}")
                # self.neptune_logger[log] = losses_n_metric_by_split
        return losses_n_metric

    def trainer(self, data_list: List[Data], model, seed: int, optimizer, pbar) -> Tuple[LossAndMetric, Any]:
        loader = DataLoader(data_list, batch_size=1, shuffle=True)

        losses_n_metric = None
        for epoch in range(self.max_epochs):
            torch.cuda.empty_cache()
            self.train(loader=loader, model=model, optimizer=optimizer)
            val_loss, val_metric = self.test(loader=loader, model=model, mask_str='val')
            test_loss, test_metric = self.test(loader=loader, model=model, mask_str='test')

            losses_n_metric = LossAndMetric(
                val_loss=val_loss, test_loss=test_loss,
                val_metric=val_metric, test_metric=test_metric
            )

            # save model in checkpoints
            if (epoch + 1) in self.checkpoints:
                tmp_args = copy.deepcopy(self.args)
                setattr(tmp_args, 'max_epochs', epoch + 1)
                check_point_path = create_model_dict_path(args=tmp_args)
                os.makedirs(check_point_path, exist_ok=True)
                model_path = osp.join(check_point_path, f"Seed{seed}.pt")
                with open(model_path, "wb") as f:
                    torch.save(model.cpu().state_dict(), f)
                model = model.to(device=self.device)

            # Record results
            for name in losses_n_metric._fields:
                losses_n_metric_by_split = getattr(losses_n_metric, name)
                log = str_print(seed=seed,
                                suffix=f"{name} epochs")
                # self.neptune_logger[log].append(losses_n_metric_by_split)

            pbar_str = str_print(train_test_name=self.train_test_setup.name, seed=seed,
                                 losses_n_metric=losses_n_metric)
            pbar.set_description(pbar_str)
            pbar.update(n=1)

        return losses_n_metric, model.cpu().state_dict()

    def train(self, loader, model, optimizer):
        model.train()
        num_examples = len(loader)
        optimizer.zero_grad()
        for data in loader:
            train_y = copy.deepcopy(data.y_mat)
            train_y[~data.train_mask] = 0
            scores, gt_mask = model(data.x, train_y=train_y,
                                    xy_conversions=data.xy_conversions, is_batch=True, device=self.device,
                                    edge_index=data.edge_index, rowptr=data.rowptr, indices=data.indices,
                                    )
            gt_mask = gt_mask.to(device=self.device)

            # loss
            train_loss = TASK_LOSS(scores[gt_mask], data.y.to(device=self.device)[gt_mask]) / num_examples

            # backward
            train_loss.backward()
        optimizer.step()

    def test(self, loader, model, mask_str: str) -> Tuple[float, float]:
        model.eval()
        loss, metric = 0, 0
        loader_size = len(loader)
        for data in loader:
            train_y = copy.deepcopy(data.y_mat)
            train_y[~data.train_mask] = 0
            scores, _ = model(data.x, train_y=train_y,
                              xy_conversions=data.xy_conversions, is_batch=False, device=self.device,
                              edge_index=data.edge_index, rowptr=data.rowptr, indices=data.indices,
                              )
            gt_mask = getattr(data, mask_str + '_mask').to(device=self.device)

            # loss
            loss += TASK_LOSS(scores[gt_mask],
                              data.y.to(device=self.device)[gt_mask]).detach().cpu().item() / loader_size

            # metric
            gt_mask = gt_mask.cpu()
            scores = scores.detach().cpu()
            if scores.shape[1] == 2:
                y_pred = scores[gt_mask][:, 1].numpy()
                y_true = data.y[gt_mask].numpy()
                metric += average_precision_score(y_true, y_pred)

            else:
                y_pred = scores[gt_mask].argmax(dim=1).numpy()
                y_true = data.y[gt_mask].numpy()
                metric += accuracy_score(y_true, y_pred)
            
        return loss, metric
