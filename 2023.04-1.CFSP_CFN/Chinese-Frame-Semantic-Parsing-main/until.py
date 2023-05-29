from torch.utils.data import DataLoader
import torch
import json
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn


class Example:
    def __init__(self, idx, lex_name, sentence, lex_pos, gloss, label):
        self.task_id = idx,
        self.lex_name = lex_name,
        self.sentence = sentence,
        self.lex_pos = lex_pos,
        self.gloss = gloss,
        self.label = label


class InputFeatures1(object):
    """A single set of features of data."""

    def __init__(self, input_ids, mask_array, graph_node, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.mask_array = mask_array
        self.graph_node = graph_node
        self.label_id = label_id


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, task_id, input_ids, mask_array, input_mask, segment_ids, label_id):
        self.task_id = task_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.mask_array = mask_array
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    @classmethod
    def _read_txt(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            data = f.readlines()
            lines = []
            for line in data:
                lines.append(line)
            return lines


class Processor(DataProcessor):
    def get_train_examples(self, data_dir, f2idx):
        """See base class."""
        train_data = []
        data = json.load(open(data_dir, encoding="utf8"))
        train_data = [{
            "task_id": item['sentence_id'],
            "frame": item['frame'],
            "lex": item['text'][ item["target"]["start"]: item["target"]["end"] + 1 ],
            "sentence": item['text'],
            "label": f2idx.get(item['frame'], 0),
            "gloss": "来了",
        } for item in data]

        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir, f2idx):
        """See base class."""
        dev_data = []
        data = json.load(open(data_dir, encoding="utf8"))
        dev_data = [{
            "task_id": item['sentence_id'],
            "frame": item['frame'],
            "lex": item['text'][ item["target"]["start"]: item["target"]["end"] + 1 ],
            "sentence": item['text'],
            "label": f2idx.get(item['frame'], 0),
            "gloss": "来了",
        } for item in data]
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir, f2idx):
        """See base class."""
        train_data = []
        data = json.load(open(data_dir, encoding="utf8"))
        train_data = [{
            "task_id": item['sentence_id'],
            "frame": item['frame'],
            "lex": item['text'][ item["target"]["start"]: item["target"]["end"] + 1 ],
            "sentence": item['text'],
            "label": f2idx.get(item['frame'], 0),
            "gloss": "来了",
        } for item in data]
        return self._create_examples(train_data, "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        # max_sen_length = 0
        for (i, line) in enumerate(lines):
            """
            line = {"label":1 ,"lex":" ",sentence:" "}
            """
            # if set_type == 'train' and i >=1000: break
            # if set_type == 'dev' and i>=500: break
            guid = "%s-%s" % (set_type, i)
            lex_name = line['lex']
            # features = torch.tensor(line['features'], dtype=torch.double)
            sentence = line['sentence']
            try:
            #     gloss = line['gloss']
            #
                # lex_pos = sentence.split().index(lex_name)
                lex_pos = sentence.index(lex_name)
                # lex_pos = 0
            except Exception as e1:
                print(e1)
                print(sentence)
                print(lex_name)
                continue
            label = line['label']
            if not isinstance(label, int):
                label = label.replace('\n', '').replace('\t', '').replace(' ', '')

            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("lex_name=", lex_name)
                print("sentence=", sentence)
                print("lex_pos=", lex_pos)

            examples.append(
                Example(idx=line['task_id'], lex_name=lex_name, sentence=sentence, lex_pos=lex_pos, gloss=None, label=label))
        # print("max_length", max_sen_length)
        # print(len(lines))
        return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        orig_tokens = example.sentence
        lex_pos = example.lex_pos[0]
        # lex_len = len(example.lex_name[0])

        # orig_tokens_list = list(orig_tokens)
        # orig_tokens_list.insert(lex_pos, '[CLS]')
        # orig_tokens_list.insert(lex_pos+lex_len+1, '[SEP]')
        # orig_tokens = "".join(orig_tokens_list)
        if lex_pos > 254:
            continue
        bert_tokens = []
        # token = tokenizer.tokenize(orig_tokens[0])
        bert_tokens.extend(tokenizer.tokenize(orig_tokens[0]))
        # lex_pos = 0
        # lex_len = 0
        # pos = 0
        lex = example.lex_name[0]
        lex_len = len(lex)
        # orig_tokens = orig_tokens[0].split(' ')
        # for idx, value in enumerate(orig_tokens):
        #     token = tokenizer.tokenize(value)
        #
        #     if value in lex:
        #         if pos > lex_pos == 0:
        #             lex_pos = pos
        #         if pos == lex_pos+lex_len:
        #             lex_len += len(token)
        #     pos += len(token)
        #     bert_tokens.extend(token)
        if len(bert_tokens) > max_seq_length - 2:
            bert_tokens = bert_tokens[:(max_seq_length - 2)]

        # bert_tokens.insert(lex_pos, '[STAR]')
        # bert_tokens.insert(lex_pos + lex_len + 1, '[ENDD]')
        # print("lex:{} bert_token_lex:{}".format(example.lex_name[0], bert_tokens[lex_pos+1:lex_pos+lex_len+1]))
        bert_tokens = ["[CLS]"] + bert_tokens + ["[SEP]"]
        segment_ids = [0] * len(bert_tokens)
        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        input_mask = [1] * len(input_ids)
        mask_array = np.zeros(256)
        mask_array[lex_pos+1:lex_pos+lex_len+1] = 1
        # mask_array = np.array([lex_pos, lex_len])
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        try:
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
        except:
            print(len(input_ids))

        label_id = example.label

        if ex_index < 5:
            print("*** Example ***")
            print("task_id: %d" % (example.task_id))
            print("lex:{}".format(example.lex_name[0]))
            print("tokens: %s" % " ".join([str(x) for x in bert_tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("mask_array: %s" % " ".join([str(x) for x in mask_array]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(
                          task_id=example.task_id,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          mask_array=mask_array,
                          ))
    return features


class Processor_chinese(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = []
        with open(data_dir) as f:
            data = f.readlines()
            for line in data:
                train_data.append(eval(line))
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = []
        with open(data_dir) as f:
            data = f.readlines()
            for line in data:
                dev_data.append(eval(line))
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        train_data = []
        with open(data_dir) as f:
            data = f.readlines()
            for line in data:
                train_data.append(eval(line))
        return self._create_examples(train_data, "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        # max_sen_length = 0
        for (i, line) in enumerate(lines):
            """
            line = {"label":1 ,"lex":" ",sentence:" "}
            """
            # if set_type == 'train' and i >=1000: break
            # if set_type == 'dev' and i>=500: break
            guid = "%s-%s" % (set_type, i)
            lex_name = line['lex']
            # features = torch.tensor(line['features'], dtype=torch.double)
            sentence = line['sentence']
            try:
            #     gloss = line['gloss']
            #
                # lex_pos = sentence.split().index(lex_name)
                lex_pos = sentence.index(lex_name)
            except Exception as e1:
                print(e1)
                print(sentence)
                print(lex_name)
                continue
            label = line['label']
            if not isinstance(label, int):
                label = label.replace('\n', '').replace('\t', '').replace(' ', '')

            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("lex_name=", lex_name)
                print("sentence=", sentence)
                print("lex_pos=", lex_pos)

            examples.append(
                Example(idx=i, lex_name=lex_name, sentence=sentence, lex_pos=lex_pos, gloss=None, label=label))
        # print("max_length", max_sen_length)
        # print(len(lines))
        return examples


q_f = open('./ccl-cfn/q.json', 'a', encoding='utf8')


def convert_examples_to_features_graph(examples, max_seq_length, tokenizer, ltp):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        orig_tokens = example.sentence
        seg, hidden = ltp.seg([orig_tokens[0]])
        srl = ltp.srl(hidden)
        try:
            srl_lex_pos = seg[0].index(example.lex_name[0])
        except:
            print(example.lex_name[0])
            q_f.write("lex:{},sentence:{}".format(example.lex_name[0], example.sentence[0]))
            q_f.write("\n")
            continue
        graph_node = []
        for detail in srl[0][srl_lex_pos]:
            # print(''.join(seg[0][detail[1]:detail[2] + 1]))
            # print(orig_tokens.index(''.join(seg[0][detail[1]:detail[2] + 1])))
            graph_node.append(orig_tokens[0].index(''.join(seg[0][detail[1]:detail[2] + 1])))
            graph_node.append(orig_tokens[0].index(''.join(seg[0][detail[1]:detail[2] + 1])) + len(
                ''.join(seg[0][detail[1]:detail[2] + 1])))

        lex_pos = example.lex_pos[0]
        lex_len = len(example.lex_name[0])
        # orig_tokens_list = list(orig_tokens)
        # orig_tokens_list.insert(lex_pos, '[CLS]')
        # orig_tokens_list.insert(lex_pos+lex_len+1, '[SEP]')
        # orig_tokens = "".join(orig_tokens_list)
        if lex_pos > 254:
            continue
        bert_tokens = []
        # token = tokenizer.tokenize(orig_tokens[0])
        bert_tokens.extend(tokenizer.tokenize(orig_tokens[0]))
        if len(bert_tokens) > max_seq_length - 4:
            bert_tokens = bert_tokens[:(max_seq_length - 4)]

        bert_tokens.insert(lex_pos, '[CLS]')
        bert_tokens.insert(lex_pos + lex_len + 1, '[SEP]')
        bert_tokens = ["[CLS]"] + bert_tokens + ["[SEP]"]
        segment_ids = [0] * len(bert_tokens)
        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        input_mask = [1] * len(input_ids)
        mask_array = np.zeros(256)
        mask_array[lex_pos + 1] = 1
        # mask_array = np.array([lex_pos, lex_len])
        padding = [0] * (max_seq_length - len(input_ids))
        padding1 = [0] * (max_seq_length - len(graph_node))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        graph_node += padding1
        try:
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
        except:
            print(len(input_ids))

        label_id = example.label

        if ex_index < 5:
            print("*** Example ***")
            print("guid: %s" % (example.guid[0]))
            print("lex:{}".format(example.lex_name[0]))
            print("tokens: %s" % " ".join([str(x) for x in bert_tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("mask_array: %s" % " ".join([str(x) for x in mask_array]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures1(input_ids=input_ids,
                           input_mask=input_mask,
                           segment_ids=segment_ids,
                           label_id=label_id,
                           mask_array=mask_array,
                           graph_node=graph_node
                           ))
    return features


# 将多个字的tensor pool成一个(,768)的tensor
def avg_pool(input_tensor):
    if input_tensor.size(0) >= 2:
        m = torch.nn.AdaptiveAvgPool2d((1, 768))
        input_tensor = input_tensor.unsqueeze(0)
        # 变成一个(,768)的tensor
        output = m(input_tensor).squeeze(0).squeeze(0)
        return output
    else:
        return input_tensor


def convert_examples_to_features_graph_gloss(examples, max_seq_length, tokenizer, ltp):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        orig_tokens = example.sentence
        gloss = example.gloss
        seg, hidden = ltp.seg([orig_tokens[0]])
        srl = ltp.srl(hidden)
        try:
            srl_lex_pos = seg[0].index(example.lex_name[0])
        except:
            print(example.lex_name[0])
            q_f.write("lex:{},sentence:{}".format(example.lex_name[0], example.sentence[0]))
            q_f.write('\n')
            continue
        graph_node = []
        for detail in srl[0][srl_lex_pos]:
            # print(''.join(seg[0][detail[1]:detail[2] + 1]))
            # print(orig_tokens.index(''.join(seg[0][detail[1]:detail[2] + 1])))
            graph_node.append(orig_tokens[0].index(''.join(seg[0][detail[1]:detail[2] + 1])))
            graph_node.append(orig_tokens[0].index(''.join(seg[0][detail[1]:detail[2] + 1])) + len(
                ''.join(seg[0][detail[1]:detail[2] + 1])))

        lex_pos = example.lex_pos[0]
        lex_len = len(example.lex_name[0])
        # orig_tokens_list = list(orig_tokens)
        # orig_tokens_list.insert(lex_pos, '[CLS]')
        # orig_tokens_list.insert(lex_pos+lex_len+1, '[SEP]')
        # orig_tokens = "".join(orig_tokens_list)
        if lex_pos > 510:
            continue
        bert_tokens = []
        gloss_tokens = []
        gloss_tokens.extend(tokenizer.tokenize(gloss[0]))
        # token = tokenizer.tokenize(orig_tokens[0])
        bert_tokens.extend(tokenizer.tokenize(orig_tokens[0]))


        bert_tokens.insert(lex_pos, '[CLS]')
        bert_tokens.insert(lex_pos + lex_len + 1, '[SEP]')
        bert_tokens = ["[CLS]"] + bert_tokens + ["[SEP]"]
        segment_ids = [0] * len(bert_tokens) + [1] * len(gloss_tokens)
        bert_tokens = bert_tokens + gloss_tokens
        if len(bert_tokens) > max_seq_length - 5:
            bert_tokens = bert_tokens[:(max_seq_length - 5)]
            segment_ids = segment_ids[:(max_seq_length - 5)]
        bert_tokens = bert_tokens + ["[SEP]"]
        segment_ids = segment_ids + [1]
        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        input_mask = [1] * len(input_ids)
        mask_array = np.zeros(512)
        mask_array[lex_pos + 1] = 1
        # mask_array = np.array([lex_pos, lex_len])
        padding = [0] * (max_seq_length - len(input_ids))
        padding1 = [0] * (max_seq_length - len(graph_node))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        graph_node += padding1
        try:
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
        except:
            print(len(input_ids))

        label_id = example.label

        if ex_index < 5:
            print("*** Example ***")
            print("guid: %s" % (example.guid[0]))
            print("lex:{}".format(example.lex_name[0]))
            print("tokens: %s" % " ".join([str(x) for x in bert_tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("mask_array: %s" % " ".join([str(x) for x in mask_array]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures1(input_ids=input_ids,
                           input_mask=input_mask,
                           segment_ids=segment_ids,
                           label_id=label_id,
                           mask_array=mask_array,
                           graph_node=graph_node
                           ))
    return features


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss


def convert_examples_to_features_dep_graph(examples, max_seq_length, tokenizer, ltp):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        orig_tokens = example.sentence
        seg, hidden = ltp.seg([orig_tokens[0]])
        dep = ltp.dep(hidden)
        try:
            dep_lex_pos = seg[0].index(example.lex_name[0]) + 1
        except:
            print(example.lex_name[0])
            q_f.write("lex:{},sentence:{}".format(example.lex_name[0], example.sentence[0]))
            q_f.write("\n")
            continue
        graph_node = []
        for idx, value in enumerate(dep[0]):
            if (value[2] == 'SBV' or value[2] == 'VOB') and value[1] == dep_lex_pos:
                head = value[0] - 1
                graph_node.append(orig_tokens[0].index(seg[0][head]))
                graph_node.append(orig_tokens[0].index(seg[0][head])+len(seg[0][head]))

        lex_pos = example.lex_pos[0]
        lex_len = len(example.lex_name[0])
        if lex_pos > 254:
            continue
        bert_tokens = []
        # token = tokenizer.tokenize(orig_tokens[0])
        bert_tokens.extend(tokenizer.tokenize(orig_tokens[0]))
        if len(bert_tokens) > max_seq_length - 4:
            bert_tokens = bert_tokens[:(max_seq_length - 4)]

        bert_tokens.insert(lex_pos, '[CLS]')
        bert_tokens.insert(lex_pos + lex_len + 1, '[SEP]')
        bert_tokens = ["[CLS]"] + bert_tokens + ["[SEP]"]
        segment_ids = [0] * len(bert_tokens)
        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        input_mask = [1] * len(input_ids)
        mask_array = np.zeros(256)
        mask_array[lex_pos + 1] = 1
        # mask_array = np.array([lex_pos, lex_len])
        padding = [0] * (max_seq_length - len(input_ids))
        padding1 = [0] * (max_seq_length - len(graph_node))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        graph_node += padding1
        try:
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
        except:
            print(len(input_ids))

        label_id = example.label

        if ex_index < 5:
            print("*** Example ***")
            print("guid: %s" % (example.guid[0]))
            print("lex:{}".format(example.lex_name[0]))
            print("tokens: %s" % " ".join([str(x) for x in bert_tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("mask_array: %s" % " ".join([str(x) for x in mask_array]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures1(input_ids=input_ids,
                           input_mask=input_mask,
                           segment_ids=segment_ids,
                           label_id=label_id,
                           mask_array=mask_array,
                           graph_node=graph_node
                           ))
    return features



