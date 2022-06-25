import cogie

# tokenize sentence into words
tokenize_toolkit = cogie.TokenizeToolkit(task='ws', language='english', corpus=None)
words = tokenize_toolkit.run('Ontario is the most populous province in Canada.')
# named entity recognition
ner_toolkit = cogie.NerToolkit(task='ner', language='english', corpus='trex')
ner_result = ner_toolkit.run(words)
# relation extraction
re_toolkit = cogie.ReToolkit(task='re', language='english', corpus='trex')
re_result = re_toolkit.run(words, ner_result)

token_toolkit = cogie.TokenizeToolkit(task='ws', language='english', corpus=None)
words = token_toolkit.run(
    'The true voodoo-worshipper attempts nothing of importance without certain sacrifices which are intended to propitiate his unclean gods.')
# frame identification
fn_toolkit = cogie.FnToolkit(task='fn', language='english', corpus=None)
fn_result = fn_toolkit.run(words)
# argument identification
argument_toolkit = cogie.ArgumentToolkit(task='fn', language='english', corpus='argument')
argument_result = argument_toolkit.run(words, fn_result)


#new_early_stop
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from cogktr import *
# from cogktr.core.evaluator import Evaluator
# from cogktr.utils.general_utils import init_cogktr,EarlyStopping
#
# device, output_path = init_cogktr(
#     device_id=3,
#     output_path="/data/hongbang/CogKTR/datapath/sentence_pair/QNLI/experimental_result/",
#     folder_tag="new_early_stop",
# )
#
# reader = QnliReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/sentence_pair/QNLI/raw_data")
# train_data, dev_data, test_data = reader.read_all()
# vocab = reader.read_vocab()
# processor = QnliProcessor(plm="bert-base-cased", max_token_len=256, vocab=vocab, debug=True)
# train_dataset = processor.process_train(train_data)
# dev_dataset = processor.process_dev(dev_data)
# test_dataset = processor.process_test(test_data)
#
# model = BaseSentencePairClassificationModel(plm="bert-base-cased", vocab=vocab)
# metric = BaseClassificationMetric(mode="binary")
# loss = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.00001)
# early_stopping = EarlyStopping(mode="max",patience=2,threshold=0.05,threshold_mode="rel",metric_name="F1")
#
# trainer = Trainer(model,
#                   train_dataset,
#                   dev_data=dev_dataset,
#                   n_epochs=100,
#                   batch_size=32,
#                   loss=loss,
#                   optimizer=optimizer,
#                   scheduler=None,
#                   metrics=metric,
#                   early_stopping=early_stopping,
#                   train_sampler=None,
#                   dev_sampler=None,
#                   drop_last=False,
#                   gradient_accumulation_steps=1,
#                   num_workers=5,
#                   print_every=None,
#                   scheduler_steps=None,
#                   # checkpoint_path="/data/hongbang/CogKTR/datapath/sentence_pair/QNLI/experimental_result/simple_test1--2022-05-30--13-02-12.95/model/checkpoint-300",
#                   validate_steps=500,  # validation setting
#                   save_steps=None,  # when to save model result
#                   output_path=output_path,
#                   grad_norm=1,
#                   use_tqdm=True,
#                   device=device,
#                   callbacks=None,
#                   metric_key=None,
#                   fp16=False,
#                   fp16_opt_level='O1',
#                   )
# trainer.train()
# print("end")
#
# # evaluator = Evaluator(
# #     model=model,
# #     checkpoint_path="/data/hongbang/CogKTR/datapath/sentence_pair/QNLI/experimental_result/simple_test1--2022-05-30--13-02-12.95/model/checkpoint-400",
# #     dev_data=dev_dataset,
# #     metrics=metric,
# #     sampler=None,
# #     drop_last=False,
# #     collate_fn=None,
# #     file_name="models.pt",
# #     batch_size=32,
# #     device=device,
# #     user_tqdm=True,
# # )
# # evaluator.evaluate()
# # print("End")
