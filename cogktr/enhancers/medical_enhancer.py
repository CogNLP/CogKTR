from cogktr.enhancers import BaseEnhancer

# from transformers import (
#     AdamW,
#     AdapterConfig,
#     AdapterType,
#     AutoConfig,
#     AutoModelForSequenceClassification,
#     AutoTokenizer,
# )

class MedicalEnhancer(BaseEnhancer):
    def __init__(self):
        super.__init__()

    def run_pretrain(self, plm):
        # model = AutoModelForSequenceClassification.from_pretrained(plm)
        pass



