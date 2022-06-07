import numpy as np
import torch
import transformers
from transformers import BertConfig
from cogktr.utils.log_utils import logger, init_logger

print("Hello World!")

init_logger()
logger.info("Hello World!")
init_logger("log.txt")
logger.info("Hello World Again!")
