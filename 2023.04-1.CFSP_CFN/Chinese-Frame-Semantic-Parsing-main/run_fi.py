import torch
import os
import json
import random
from torch import nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from until import Processor, convert_examples_to_features, FocalLoss
from pytorch_pretrained_bert.modeling import BertForTokenClassification2
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertAdam
import numpy as np
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
train_batch_size = 32
dev_batch_size = 64
test_batch_size = 64
learning_rate = 0.0001

epochs = 45
random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)
# device = 'cpu'
device = 'cuda:0'
do_train = True
do_dev = True
processor = Processor()
train_data_dir = './ccl-cfn/cfn-train.json'
dev_data_dir = './ccl-cfn/cfn-dev.json'
test_data_dir = './ccl-cfn/cfn-test.json'
save_dir = 'model_save/'
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model_new_test.bin"
frame_data = json.load(open("./ccl-cfn/frame_info.json", encoding="utf8"))
idx2f = [ x['frame_name'] for x in frame_data ]
f2idx = { x['frame_name']:i for i, x in enumerate(frame_data) }
num_labels = len(f2idx)
model_name = "./bert_wwm"
model = BertForTokenClassification2.from_pretrained(model_name, cache_dir=None, num_labels=num_labels)
model.to(device)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]


tokenizer = BertTokenizer.from_pretrained(model_name)
focal_loss = FocalLoss()
criteon = nn.CrossEntropyLoss().to(device)
# train_input_ids = torch.LongTensor()
train_data_processor = processor.get_train_examples(train_data_dir, f2idx)
train_feature = convert_examples_to_features(train_data_processor, 256, tokenizer)
train_task_id = torch.tensor([f.task_id for f in train_feature], dtype=torch.long)
train_input_ids = torch.tensor([f.input_ids for f in train_feature], dtype=torch.long)
train_input_mask = torch.tensor([f.input_mask for f in train_feature], dtype=torch.long)
train_segment_ids = torch.tensor([f.segment_ids for f in train_feature], dtype=torch.long)
# train_mask_array = torch.tensor([f.mask_array for f in train_feature], dtype=torch.int)
train_mask_array = torch.tensor([f.mask_array for f in train_feature], dtype=torch.bool)

train_label = torch.tensor([f.label_id for f in train_feature], dtype=torch.long)
train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_mask_array, train_label)
train_loader = DataLoader(train_data, shuffle=False, batch_size=train_batch_size)


num_train_optimization_steps = int(
            len(train_data_processor) / train_batch_size / 1) * epochs
optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate, warmup=0.1, t_total=num_train_optimization_steps)
scheduler_args={f'gamma': .75**(1 / 5090)}
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,**scheduler_args)
dev_data_processor = processor.get_dev_examples(dev_data_dir, f2idx)
dev_feature = convert_examples_to_features(dev_data_processor, 256, tokenizer)
dev_task_id = torch.tensor([f.task_id for f in dev_feature], dtype=torch.long)
dev_input_ids = torch.tensor([f.input_ids for f in dev_feature], dtype=torch.long)
dev_input_mask = torch.tensor([f.input_mask for f in dev_feature], dtype=torch.long)
dev_segment_ids = torch.tensor([f.segment_ids for f in dev_feature], dtype=torch.long)
# dev_mask_array = torch.tensor([f.mask_array for f in dev_feature], dtype=torch.int)
dev_mask_array = torch.tensor([f.mask_array for f in dev_feature], dtype=torch.bool)

dev_label = torch.tensor([f.label_id for f in dev_feature], dtype=torch.long)
dev_data = TensorDataset(dev_input_ids, dev_input_mask, dev_segment_ids, dev_mask_array, dev_label)
dev_loader = DataLoader(dev_data, shuffle=False, batch_size=dev_batch_size)


def do_test():
    test_data_processor = processor.get_test_examples(test_data_dir, f2idx)
    test_feature = convert_examples_to_features(test_data_processor, 256, tokenizer)
    test_task_id = torch.tensor([f.task_id for f in test_feature], dtype=torch.long)
    test_input_ids = torch.tensor([f.input_ids for f in test_feature], dtype=torch.long)
    test_input_mask = torch.tensor([f.input_mask for f in test_feature], dtype=torch.long)
    test_segment_ids = torch.tensor([f.segment_ids for f in test_feature], dtype=torch.long)
    # dev_mask_array = torch.tensor([f.mask_array for f in dev_feature], dtype=torch.int)
    test_mask_array = torch.tensor([f.mask_array for f in test_feature], dtype=torch.bool)

    test_label = torch.tensor([f.label_id for f in test_feature], dtype=torch.long)
    test_data = TensorDataset(test_task_id, test_input_ids, test_input_mask, test_segment_ids, test_mask_array, test_label)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=test_batch_size)

    res_json = []
    model.eval()
    for step, batch in enumerate(tqdm(test_loader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        test_task_id, input_ids, input_mask, segment_ids, mask_array, label = batch
        with torch.no_grad():
            logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None,
                    mask_array=mask_array)

        logits_ = F.softmax(logits, dim=-1)
        pred = torch.cat([test_task_id, logits_.argmax(-1).unsqueeze(-1)], -1)
        res_json.extend(pred.tolist())

    json.dump([ [x[0], idx2f[x[1]]] for x in res_json], open("./ccl-cfn/result/task1_test.json", "w",encoding="utf8"), ensure_ascii=False)


best_acc = 0.0
dev_loss = []
for epoch in range(epochs):
    print("epoch:", epoch)
    if do_train:
        model.train()
        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, mask_array, label = batch
            logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None,
                           mask_array=mask_array)

            loss = criteon(logits.view(-1, num_labels), label.view(-1))
            # loss = focal_loss(logits, label)
            logits_ = F.softmax(logits, dim=-1)
            logits_ = logits_.detach().cpu().numpy()
            outputs = np.argmax(logits_, axis=1)
            batch_acc = np.sum(outputs == label.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % 100 == 0:
                print('Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc:{}'.format(
                    epoch, step * len(batch), len(train_loader),
                           100. * step / len(train_loader), loss.item(), batch_acc/train_batch_size))

    if do_dev:
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        for step, batch in enumerate(tqdm(dev_loader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, mask_array, label = batch
            with torch.no_grad():
                logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None,
                           mask_array=mask_array)

                loss = criteon(logits, label)
            logits_ = F.softmax(logits, dim=-1)
            outputs = np.argmax(logits_.detach().cpu().numpy(), axis=1)
            batch_acc = np.sum(outputs == label.detach().cpu().numpy())
            eval_loss += loss
            eval_accuracy += batch_acc
        eval_acc = eval_accuracy / len(dev_loader)
        eval_loss = eval_loss / len(dev_loader)
        print("eval_acc:{}".format(eval_acc/dev_batch_size))
        print("eval_loss:{}".format(eval_loss))
        if eval_acc > best_acc:
            best_acc = eval_acc
            print("best_acc:{}".format(best_acc/dev_batch_size))
            torch.save(model, save_dir+WEIGHTS_NAME+str(epoch))
            do_test()
        dev_loss.append(eval_loss)
print("best_acc:{}".format(best_acc/dev_batch_size))
print(dev_loss)




