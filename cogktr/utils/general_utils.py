from cogktr.utils.log_utils import init_logger
import random
import os
import torch
import numpy as np
import datetime
import torch.distributed as dist


def move_dict_value_to_device(batch, device, rank=-1, non_blocking=False):
    if rank == -1 and not isinstance(device, torch.device):
        raise TypeError(f"device must be `torch.device` in a single process, got `{type(device)}`")

    _move_dict_value_to_device(batch, device, rank, non_blocking)


def _move_dict_value_to_device(batch, device, rank=-1, non_blocking=False):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if rank == -1:
                batch[key] = value.to(device, non_blocking=non_blocking)
            else:
                batch[key] = value.to(torch.device("cuda:{}".format(rank)), non_blocking=non_blocking)
        elif isinstance(value, dict):
            _move_dict_value_to_device(batch[key], device, rank, non_blocking)


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / nprocs
    return rt


class EarlyStopping:
    def __init__(self,mode="min",patience=4,threshold=1e-4,threshold_mode="rel",metric_name=None):
        if mode not in ["min","max"]:
            assert ValueError("Argument mode must be min or max but got {}".format(mode))
        if threshold_mode not in ["rel","abs"]:
            assert ValueError("Argument threshold_mode must be rel or abs but got {}".format(threshold_mode))
        if threshold_mode == 'rel' and (threshold < 0 or threshold > 1):
            assert ValueError("Threshold must be in [0,1] in rel threshold mode but got {}.".format(threshold))
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode

        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.metric_name = metric_name

    def __call__(self, value):
        if self.best_value == None:
            self.best_value = value
            return

        if self.is_better(value,self.best_value):
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


    def is_better(self,value,best_value):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return value < best_value * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return value < best_value - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return value > best_value * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return value > best_value + self.threshold


def init_cogktr(
        output_path,
        device_id=None,
        folder_tag="",
        seed=1,
        rank=-1,
):
    # set the random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not isinstance(device_id, int):
        if rank == -1:
            raise ValueError("Device Id can noly be int in a single process!")

    if rank in [0, -1]:
        # calculate the output path
        new_output_path = os.path.join(
            output_path,
            folder_tag + "--" + str(datetime.datetime.now())[:-4].replace(':', '-').replace(' ', '--')
        )
        if not os.path.exists(os.path.abspath(new_output_path)):
            os.makedirs(os.path.abspath(new_output_path))

        # initialize the logger configuration
        init_logger(os.path.join(new_output_path, "log.txt"), rank=rank)

        # set the cuda device
        if rank == -1:
            device = torch.device('cuda:{}'.format(device_id) if torch.cuda.is_available() == True else "cpu")
        else:
            device = torch.device("cuda:0")
    else:
        new_output_path = None
        init_logger(new_output_path, rank=rank)
        device = torch.device("cuda:0")

    return device, new_output_path
