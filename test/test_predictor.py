import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.general_utils import init_cogktr
from cogktr.data.processor.openbookqa_processors.openbookqa_processor import OpenBookQAProcessor
from cogktr.core.predictor import Predictor

device, output_path = init_cogktr(
    device_id=5,
    output_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/experimental_result/",
    folder_tag="debug_predictor",
)

reader = OpenBookQAReader(
    raw_data_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/raw_data")
train_data, dev_data, test_data = reader.read_all()
vocab = reader.read_vocab()

processor = OpenBookQAProcessor(
    plm="roberta-large",
    max_token_len=100,
    vocab=vocab,
    debug=True,
)

test_dataset = processor.process_test(test_data)

plm = PlmAutoModel(pretrained_model_name="roberta-large")
model = BaseQuestionAnsweringModel(plm=plm, vocab=vocab)

predictor = Predictor(
    model=model,
    checkpoint_path="/data/hongbang/CogKTR/datapath/question_answering/OpenBookQA/experimental_result/test_obqa_base--2022-07-31--14-36-59.98/best_model/checkpoint-8432",
    dev_data=test_dataset,
    device=device,
    vocab=vocab,
    batch_size=2,
)
results = predictor.predict()
num_choice = 4
target_data = test_data
for i,result in enumerate(results):
    idx = i * num_choice
    stem = target_data["stem"][idx]
    answerKey = target_data["answerKey"][idx]
    answers = ["{}:{}".format(target_data["key"][tmp_idx],target_data["answer_text"][tmp_idx])
               for tmp_idx in range(i * num_choice,(i+1) * num_choice)]
    print(stem)
    print("\n".join(answers))
    print("AnswerKey:{}  Prediction:{}".format(answerKey,result))
    print("-------------------")
