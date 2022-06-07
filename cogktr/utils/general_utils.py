from cogktr.utils.log_utils import init_logger
import random
import os
import torch
import numpy as np
import datetime
import torch.distributed as dist

def move_dict_value_to_device(batch,device,rank=-1,non_blocking=False):
    if rank == -1 and not isinstance(device, torch.device):
        raise TypeError(f"device must be `torch.device` in a single process, got `{type(device)}`")

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if rank == -1:
                batch[key] = value.to(device)
            else:
                batch[key] = value.to(torch.device("cuda:{}".format(rank)))


def reduce_mean(tensor,nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt,op=dist.ReduceOp.SUM)
    rt = rt/nprocs
    return rt

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

    if not isinstance(device_id,int):
        if rank == -1:
            raise ValueError("Device Id can noly be int in a single process!")

    if rank in [0,-1]:
        # calculate the output path
        new_output_path = os.path.join(
            output_path,
            folder_tag + "--" + str(datetime.datetime.now())[:-4].replace(':', '-').replace(' ', '--')
        )
        if not os.path.exists(os.path.abspath(new_output_path)):
            os.makedirs(os.path.abspath(new_output_path))

        # initialize the logger configuration
        init_logger(os.path.join(new_output_path,"log.txt"),rank=rank)

        # set the cuda device
        if rank == -1:
            device = torch.device('cuda:{}'.format(device_id) if torch.cuda.is_available() == True else "cpu")
        else:
            device = torch.device("cuda:0")
    else:
        new_output_path = None
        init_logger(new_output_path,rank=rank)
        device = torch.device("cuda:0")

    return device,new_output_path
