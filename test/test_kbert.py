import torch
import torch.nn as nn
import torch.optim as optim
from cogktr import *
from cogktr.utils.general_utils import init_cogktr
from cogktr.utils.constant.kbert_constants.constants import *
from cogktr.data.processor.sst2_processors.sst2_for_kbert_processor import *
from cogktr.models.kbert_model import KBertModelBuilder
from cogktr.modules.layers.optimizers import BertAdam

# initiate
device, output_path = init_cogktr(
    device_id=3,
    output_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/text_classification/SST_2/experimental_result",
    folder_tag="simple_test",
)

# Load the data
reader = Sst2Reader(raw_data_path="/home/chenyuheng/zhouyuyang/CogKTR/datapath/text_classification/SST_2/raw_data")
train_data, dev_data, test_data = reader.read_all()

# Load vocabulary
vocab = Vocabulary()
vocab.load("/home/chenyuheng/zhouyuyang/CogKTR/cogktr/utils/constant/kbert_constants/vocab.txt")

# build knowledgegraph
kg = KnowledgeGraph(spo_file_paths=["/home/chenyuheng/zhouyuyang/CogKTR/datapath/knowledge_graph/wikidata/wikidata.spo"],
                        predicate=True)

# process data
processor = Sst2ForKbertProcessor()
train_dataset = processor.process_train([train_data, kg, vocab, 256])
dev_dataset = processor.process_dev([dev_data, kg, vocab, 256])

# build model
modelbuilder = KBertModelBuilder(emb_size=768,
                                 vocab=vocab,
                                 layers_num=12,
                                 hidden_size=768,
                                 heads_num=12,
                                 feedforward_size=3072,
                                 dropout=0.1,
                                 )
modelbuilder.load_params("/home/chenyuheng/zhouyuyang/K-Bert/models/google_model.bin")
modelbuilder.build_classification_model()
model = modelbuilder.model

# Train the data
print("Trans data to tensor.")
print("input_ids")
input_ids = torch.LongTensor(train_dataset.datable.datas["token_ids"])
print("label_ids")
label_ids = torch.LongTensor(train_dataset.datable.datas["label"])
print("mask_ids")
mask_ids = torch.LongTensor(train_dataset.datable.datas["mask"])
print("pos_ids")
pos_ids = torch.LongTensor(train_dataset.datable.datas["pos"])
print("vms")
vms = train_dataset.datable.datas["vm"]

instances_num = train_dataset.length
epochs_num = 5
batch_size = 32
learning_rate = 2e-5
warmup = 0.1
train_steps = int(instances_num * epochs_num / batch_size) + 1

print("Batch size: ", batch_size)
print("The number of training instances:", instances_num)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
]
optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate, warmup=warmup, t_total=train_steps)

total_loss = 0.
result = 0.0
best_result = 0.0

for epoch in range(1, epochs_num + 1):
    model.train()
    for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(
            modelbuilder.batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):
        model.zero_grad()

        vms_batch = torch.LongTensor(vms_batch)

        input_ids_batch = input_ids_batch.to(device)
        label_ids_batch = label_ids_batch.to(device)
        mask_ids_batch = mask_ids_batch.to(device)
        pos_ids_batch = pos_ids_batch.to(device)
        vms_batch = vms_batch.to(device)

        loss, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos=pos_ids_batch, vm=vms_batch)
        if torch.cuda.device_count() > 1:
            loss = torch.mean(loss)
        total_loss += loss.item()
        if (i + 1) % 100 == 0:
            print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,
                                                                              total_loss / 100))
            sys.stdout.flush()
            total_loss = 0.
        loss.backward()
        optimizer.step()

    print("Start evaluation on dev dataset.")
    result = modelbuilder.evaluate(dev_dataset, False)
    if result > best_result:
        best_result = result
        save_model(model, output_path)
    else:
        continue






