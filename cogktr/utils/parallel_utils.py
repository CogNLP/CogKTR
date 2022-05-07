import torch
import sys

def module2parallel(module, device_ids):
    if len(device_ids) == 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module.to(device)
        return module
    elif len(device_ids) == 1:
        device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
        module.to(device)
        return module
    else:
        device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
        module.to(device)
        module_name = type(module).__name__
        module = getattr(sys.modules[module.__module__], module_name + "Parallel")(module, device_ids)
        return module