import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
import logging
from cogktr.utils.io_utils import load_model


class Evaluator:
    def __init__(
            self,
            model,
            checkpoint_path,
            dev_data,
            metrics,
            sampler=None,
            collate_fn=None,
            drop_last=False,
            file_name="models.pt",
            batch_size=32,
            device="cpu",
            user_tqdm=True,
    ):


        self.model = model
        self.checkpoint_path = checkpoint_path
        self.dev_data = dev_data
        self.metrics = metrics
        self.sampler = sampler
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.file_name = file_name
        self.batch_size = batch_size
        self.device = device
        self.use_tqdm = user_tqdm

        self.dev_dataloader = DataLoader(dataset=self.dev_data, batch_size=self.batch_size,
                                         sampler=self.sampler, drop_last=self.drop_last,
                                         collate_fn=self.collate_fn)

        model_file = os.path.abspath(os.path.join(self.checkpoint_path,file_name))
        if os.path.isfile(model_file):
            self.model = load_model(self.model,model_file)
        else:
            raise ValueError("Pretrained model file {} does not exist!".format(model_file))

        self.model.to(self.device)
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)


    def evaluate(self):
        self.logger.info("Start Evaluating...")
        self.logger.info("Start time = %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        self.model.eval()
        if self.use_tqdm:
            progress = enumerate(tqdm(self.dev_dataloader, desc="Evaluating", leave=False), 1)
        else:
            progress = enumerate(self.dev_dataloader, 1)
        with torch.no_grad():
            for step, batch in progress:
                self.model.evaluate(batch, self.metrics)
        evaluate_result = self.metrics.get_metric()
        self.logger.info("Evaluate result = %s", str(evaluate_result))
        self.logger.info("End time = %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
