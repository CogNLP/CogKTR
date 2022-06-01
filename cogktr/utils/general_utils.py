from cogktr.utils.log_utils import init_logger
import random
import os
import torch
import numpy as np
import datetime


def init_cogktr(
        device_id,
        output_path,
        folder_tag="",
        seed=1,
):
    # set the random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # calculate the output path
    new_output_path = os.path.join(
        output_path,
        folder_tag + "--" + str(datetime.datetime.now())[:-4].replace(':', '-').replace(' ', '--')
    )
    if not os.path.exists(os.path.abspath(new_output_path)):
        os.makedirs(os.path.abspath(new_output_path))

    # initialize the logger configuration
    init_logger(os.path.join(new_output_path,"log.txt"))

    # set the gpu or cpu device
    device_list = str(device_id).strip().lower().replace('cuda:', '')
    cpu = device_list == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device_list:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device_list  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device_list} requested'  # check availability
    device = torch.device('cuda' if torch.cuda.is_available() == True else "cpu")

    return device,new_output_path
