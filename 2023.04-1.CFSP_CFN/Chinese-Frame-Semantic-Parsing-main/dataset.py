from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from transformers import BertTokenizerFast, PreTrainedTokenizerBase, DataCollatorWithPadding
from tqdm.auto import tqdm
from typing import Optional, Union
import json

class FrameAIDataset(Dataset):
    def __init__(self, data_file, tokenizer):
        super(FrameAIDataset, self).__init__()
        # print('load data...')
        data_instance_dic = {}
        with open(data_file, 'r') as f:
            data_instance_dic = json.load(f)
        # data_instance_dic = np.load(data_file, allow_pickle=True).item()
        self.data = []
        # self.model_name_or_path = model_name_or_path
        # print('load tokenizer...')
        self.tokenizer = tokenizer
        # self.tokenizer.add_tokens(['<f>', '</f>', '<r>', '</r>', '<t>', '</t>'])
        # bar = tqdm(range(len(data_instance_dic)))
        cnt = 0
        for item in data_instance_dic:
            # for k, v in item.items():
            #     print(k, v)
            self.tokenize_instance(item)
            # exit(0)


    def align_labels_with_tokens(self, labels, word_ids):
        new_labels = []
        current_word = -1
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id == -1 else labels[word_id]
                new_labels.append(label)
            elif word_id == -1:
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels

    def tokenize_instance(self, dic):
        # for k, v in dic.items():
        #     print(k, v)
        data_dic = {}
        task_id = dic['sentence_id']
        data_dic['task_id'] = [task_id]
        context = list(dic['text'])
        context[dic['target']['start']] = '<t> ' + context[dic['target']['start']]
        context[dic['target']["end"]] = context[dic['target']["end"]] + ' </t>'
        labels = [0]*len(context)
        for span in dic['cfn_spans']:
            labels[span['start']] = 1
            for i in range(span['start'] + 1, span['end']+1):
                labels[i] = 2
        # print(query)
        encodings = self.tokenizer(context,is_split_into_words=True, return_length=True)
        data_dic['input_ids'] = encodings['input_ids']
        data_dic['attention_mask'] = encodings['attention_mask']
        data_dic['token_type_ids'] = encodings['token_type_ids']
        data_dic['length'] = encodings['length']
        # data_dic['context_length'] = [data_dic['length'][0] - sum(data_dic['token_type_ids']) - 1]
        data_dic['word_ids'] = [x if x is not None else -1 for x in encodings.word_ids()]
        data_dic['labels'] = self.align_labels_with_tokens(labels, data_dic['word_ids'])
        while len(data_dic['word_ids']) < 512:
            data_dic['word_ids'].append(-1)
        # assert len(data_dic['labels']) == len(data_dic['input_ids'])
        # assert len(data_dic['word_ids']) == len(data_dic['input_ids'])
        # FE_token_idx_start = [encodings.word_to_tokens(x, sequence_index=1).start for x in dic['frame_data']['FE_word_idx']]
        # FE_token_idx_end = [encodings.word_to_tokens(x, sequence_index=1).end - 1 for x in dic['frame_data']['FE_word_idx']]
        # FE_token_idx_start = [encodings.word_to_tokens(x, sequence_index=1).start - 1 for x in dic['FE_word_idx']]
        # FE_token_idx_end = [encodings.word_to_tokens(x, sequence_index=1).end for x in dic['FE_word_idx']]
        # FE_token_idx = [[s, t] for s, t in zip(FE_token_idx_start, FE_token_idx_end)]
        # data_dic['FE_num'] = [len(FE_token_idx)]
        # data_dic['FE_token_idx'] = FE_token_idx
        # data_dic['start_positions'] = [encodings.word_to_tokens(x-1, sequence_index=0).start if x > 0 else 0 for x in start_positions]
        # data_dic['end_positions'] = [encodings.word_to_tokens(x-1, sequence_index=0).end - 1 if x > 0 else 0 for x in end_positions]
        # data_dic['gt_FE_word_idx'] = dic['gt_FE_word_idx']
        # data_dic['gt_start_positions'] = dic['gt_start_positions']
        # data_dic['gt_end_positions'] = dic['gt_end_positions']
        # data_dic['FE_core_pts'] = dic['FE_core_pts']
        self.data.append(data_dic)
        # for k, v in data_dic.items():
        #     print(k, v)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def subset(self, indices):
        return Subset(self, indices=indices)

class FrameRCDataset(Dataset):
    def __init__(self, data_file, tokenizer, fe2id, task1_res=None, task2_res=None):
        super(FrameRCDataset, self).__init__()
        # print('load data...')
        data_instance_dic = {}
        with open(data_file, 'r') as f:
            data_instance_dic = json.load(f)
        # data_instance_dic = np.load(data_file, allow_pickle=True).item()
        self.data = []
        if task1_res is not None:
            tid2frame = {}
            with open(task1_res, 'r') as f:
                task1_data = json.load(f)
                for item in task1_data:
                    tid2frame[item[0]] = item[1]
            for item in data_instance_dic:
                item['pred_frame'] = tid2frame[item['sentence_id']]
        if task2_res is not None:
            tid2spans = {}
            tid2spansets = {}
            with open(task2_res, 'r') as f:
                task2_data = json.load(f)
                for item in task2_data:
                    tid2spans.setdefault(item[0], [])
                    tid2spansets.setdefault(item[0], set())
                    if (item[1], item[2]) not in tid2spansets[item[0]]:
                        tid2spans[item[0]].append({ "start":item[1],"end":item[2] })
                        tid2spansets[item[0]].add((item[1], item[2]))
            for item in data_instance_dic:
                if item['sentence_id'] in tid2spans:
                    item['pred_spans'] = tid2spans[item['sentence_id']]
                else:
                    item['pred_spans'] = []

        for i, item in enumerate(data_instance_dic):
            data_instance_dic[i]['labels'] = []
            for span in item['cfn_spans']:
                data_instance_dic[i]['labels'].append(fe2id[span['fe_name']])
        # self.model_name_or_path = model_name_or_path
        # print('load tokenizer...')
        self.tokenizer = tokenizer
        # self.tokenizer.add_tokens(['<f>', '</f>', '<r>', '</r>', '<t>', '</t>'])
        # bar = tqdm(range(len(data_instance_dic)))
        cnt = 0
        for item in data_instance_dic:
            # for k, v in item.items():
            #     print(k, v)
            self.tokenize_instance(item)
            # exit(0)


    def tokenize_instance(self, dic):
        # for k, v in dic.items():
        #     print(k, v)
        data_dic = {}
        task_id = dic['sentence_id']
        data_dic['task_id'] = [task_id]
        context = list(dic['text'])
        context[dic['target']['start']] = '<t> ' + context[dic['target']['start']]
        context[dic['target']['end']] = context[dic['target']['end']] + ' </t>'
        frame_key = 'pred_frame' if 'pred_frame' in dic else 'frame'
        context[dic['target']['start']] = context[dic['target']['start']] + '<f> ' + dic[frame_key] + ' </f>'
        span_key = 'pred_spans' if 'pred_spans' in dic else 'cfn_spans'
        start_positions = []
        end_positions = []
        for span in dic[span_key]:
            context[span['start']] = '<a> ' + context[span['start']]
            start_positions.append(span['start'])
            context[span['end']] = context[span['end']] + ' </a>'
            end_positions.append(span['end'])
        # print(query)
        encodings = self.tokenizer(context,is_split_into_words=True, return_length=True)
        data_dic['input_ids'] = encodings['input_ids']
        data_dic['attention_mask'] = encodings['attention_mask']
        data_dic['token_type_ids'] = encodings['token_type_ids']
        data_dic['length'] = encodings['length']
        # data_dic['context_length'] = [data_dic['length'][0] - sum(data_dic['token_type_ids']) - 1]
        word_ids = [x if x is not None else -1 for x in encodings.word_ids()]
        data_dic['labels'] = dic['labels']
        while len(data_dic['labels']) < 16:
            data_dic['labels'].append(-100)
        # assert len(data_dic['labels']) == len(data_dic['input_ids'])
        # assert len(data_dic['word_ids']) == len(data_dic['input_ids'])
        # FE_token_idx_start = [encodings.word_to_tokens(x, sequence_index=1).start for x in dic['frame_data']['FE_word_idx']]
        # FE_token_idx_end = [encodings.word_to_tokens(x, sequence_index=1).end - 1 for x in dic['frame_data']['FE_word_idx']]
        # FE_token_idx_start = [encodings.word_to_tokens(x, sequence_index=1).start - 1 for x in dic['FE_word_idx']]
        # FE_token_idx_end = [encodings.word_to_tokens(x, sequence_index=1).end for x in dic['FE_word_idx']]
        # FE_token_idx = [[s, t] for s, t in zip(FE_token_idx_start, FE_token_idx_end)]
        # data_dic['FE_num'] = [len(start_positions)]
        # data_dic['FE_token_idx'] = FE_token_idx
        token_start_positions = [encodings.word_to_tokens(x, sequence_index=0).start for x in start_positions]
        token_end_positions = [encodings.word_to_tokens(x, sequence_index=0).end - 1 for x in end_positions]
        while len(token_start_positions) < 16:
            token_start_positions.append(0)
        while len(token_end_positions) < 16:
            token_end_positions.append(0)
        span_token_idx = [[s, t] for s, t in zip(token_start_positions, token_end_positions)]
        data_dic['span_token_idx'] = span_token_idx

        # data_dic['gt_FE_word_idx'] = dic['gt_FE_word_idx']
        # data_dic['gt_start_positions'] = dic['gt_start_positions']
        # data_dic['gt_end_positions'] = dic['gt_end_positions']
        # data_dic['FE_core_pts'] = dic['FE_core_pts']
        self.data.append(data_dic)
        # for k, v in data_dic.items():
        #     print(k, v)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def subset(self, indices):
        return Subset(self, indices=indices)
    


if __name__ == '__main__':
    # print(frame_data[1])
    # exit(0)
    tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-bert-wwm-ext')
    tokenizer.add_tokens(['<t>', '</t>', '<f>', '</f>', '<a>', '</a>'])
    # tokenizer.add_tokens(['<t>', '</t>'])
    with open('./ccl-cfn/frame_data.json', 'r') as f:
        data = json.load(f)
    fe2id = {}
    cnt = 0
    for frame in data:
        for fe in frame['fes']:
            name = fe['fe_abbr']
            if name not in fe2id:
                fe2id[name] = cnt
                cnt += 1
    d = FrameRCDataset('./ccl-cfn/cfn-dev.json', tokenizer, fe2id, './ccl-cfn/result/task1_dev.json', './ccl-cfn/result/task2_dev.json')
    dd = Subset(d, range(8))
    dc = DataCollatorWithPadding(tokenizer)
    dl = DataLoader(dd, 4, shuffle=False, collate_fn=dc)
    # print('hi')
    # print(len(dl))
    for b in dl:
        for k, v in b.items():
            print(k, v)
        # break
