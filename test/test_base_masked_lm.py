from cogktr import *

device, output_path = init_cogktr(
    device_id=3,
    output_path="/data/mentianyi/code/CogKTR/datapath/masked_language_model/LAMA/experimental_result/",
    folder_tag="simple_test",
)

reader = LamaReader(raw_data_path="/data/mentianyi/code/CogKTR/datapath/masked_language_model/LAMA/raw_data")
test_data = reader.read_all(dataset_name="google_re", isSplit=False)
vocab = reader.read_vocab()

processor = LamaProcessor(plm="bert-base-cased", max_token_len=128, vocab=vocab, debug=True)
test_dataset = processor.process_test(test_data)

plm = PlmAutoModel(pretrained_model_name="bert-base-cased")
model = BaseMaskedLM(plm=plm, vocab=vocab)
metric = BaseMaskedLMMetric(topk=5)

evaluator = Evaluator(model=model,
                      dev_data=test_dataset,
                      metrics=metric,
                      sampler=None,
                      collate_fn=None,
                      drop_last=False,
                      batch_size=32,
                      device="cpu",
                      user_tqdm=True)
evaluator.evaluate()
print("end")
