from cogktr.modules.plms.plm_auto import PlmAutoModel
from cogktr.core.metric.base_question_answering_metric import BaseQuestionAnsweringMetric
from cogktr.data.processor.s20rel_processors.s20rel_mop_processor import S20relMopProcessor
from cogktr.data.reader.s20rel_reader import S20relReader
from cogktr.enhancers.base_enhancer import BaseEnhancer
from cogktr.utils.general_utils import init_cogktr
from transformers import AdamW, AdapterConfig, AdapterType, AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

import torch.nn as nn
from cogktr.core.trainer import Trainer

class MedicalEnhancer(BaseEnhancer):
    def __init__(self, n_partition):
        super().__init__()
        self.n_partition = n_partition

    def run_pretrain(self, plm, non_sequential=True):
        model, optimizer = self._init_model(plm)
        s20rel_reader = S20relReader("/home/chenyuheng/zhouyuyang/CogKTR/datapath/knowledge_graph/S20Rel")
        s20rel_processor = S20relMopProcessor(plm="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                                              max_token_len=128)
        metric = BaseQuestionAnsweringMetric()
        loss = nn.CrossEntropyLoss()

        for group_idx in range(self.n_partition):
            if group_idx != 0 and non_sequential:
                model, optimizer = self._init_model(plm)

            train_data = s20rel_reader.read_train(group_idx)
            train_dataset = s20rel_processor.process_train(train_data)

            device, output_path = init_cogktr(
                device_id=0,
                output_path=f"/home/chenyuheng/zhouyuyang/CogKTR/datapath/knowledge_graph/S20Rel/pretrained_model/{group_idx}",
                folder_tag="mop_test",
            )

            trainer = Trainer(model,
                              train_dataset,
                              n_epochs=20,
                              batch_size=50,
                              loss=loss,
                              optimizer=optimizer,
                              scheduler=None,
                              metrics=metric,
                              train_sampler=None,
                              dev_sampler=None,
                              drop_last=False,
                              gradient_accumulation_steps=1,
                              num_workers=5,
                              print_every=None,
                              scheduler_steps=None,
                              validate_steps=100,
                              save_steps=100,
                              output_path=output_path,
                              grad_norm=1,
                              use_tqdm=True,
                              device=device,
                              callbacks=None,
                              metric_key=None,
                              fp16=False,
                              fp16_opt_level='O1',
                              collate_fn=train_dataset.to_dict,
                              )

            model.classifier = nn.Linear(
                in_features=768,
                out_features=s20rel_reader.num_class_list[group_idx],
                bias=True,
            )
            model.classifier.to(device)
            trainer.train()

    def _init_model(self, plm):
        config = AutoConfig.from_pretrained(plm)
        model = AutoModel.from_pretrained(plm, from_tf=False, config=config)
        # model = PlmAutoModel(plm)
        # model = AutoModelForSequenceClassification.from_pretrained(plm, from_tf=False, config=config)
        # adapter_config = AdapterConfig.load(
        #     "pfeiffer",
        #     reduction_factor=8,
        #     leave_out=[]
        # )
        model.add_adapter(
            "entity_predict",
            # AdapterType.text_task,
            # config=adapter_config
        )
        model.train_adapter(["entity_predict"])

        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=0.001,
            weight_decay=0.01,
            correct_bias=False,
        )

        return model, optimizer

if __name__ == "__main__":
    enhancer = MedicalEnhancer(n_partition=20)
    enhancer.run_pretrain(plm="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

