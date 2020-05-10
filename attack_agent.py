# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import copy
import logging
import random
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import nltk
from utils.Error_all import Errors
from utils.statistic_all import stat, dis
from pattern3.en import conjugate, lemma, lexeme
from nltk.corpus import wordnet
import inflection
from tqdm import tqdm

def make_instance(text_a, text_b, label, tokenizer, config, label_map):
    max_length = config.max_length
    if type(text_a) == list:
        text_a = ' '.join(text_a)
    if type(text_b) == list:
        text_b = ' '.join(text_b)
    inputs = tokenizer.encode_plus(
        text_a,
        text_b,
        add_special_tokens=True,
        max_length=max_length,
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)

    input_ids = input_ids + ([0] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

    label = label_map[label]
    input_ids, attention_mask, token_type_ids, label = torch.tensor([input_ids], dtype=torch.long), torch.tensor([attention_mask], dtype=torch.long), torch.tensor([token_type_ids], dtype=torch.long), torch.tensor([label], dtype=torch.long)
    return input_ids, attention_mask, token_type_ids, label


def adversarial_scoring(config, tokenizer, model, device, eval_examples, label_list):
    model.eval()
    label2idx = {label: i for i, label in enumerate(label_list)}
    idx2label = {i: label for i, label in enumerate(label_list)}
    is_single = (config.data_sign == 'SST-2')
    is_first_sent = (config.data_sign == 'SST-2' or config.data_sign == 'MRPC')
    with open(config.pos_file, 'w') as fd:
        for step, example in tqdm(enumerate(eval_examples), desc="Adversarial scoring"):
            text_a = example.text_a.strip().split()
            if is_single:
                text_b = example.text_b
            else:
                text_b = example.text_b.strip().split()
            label = example.label
            logit_lst = []
            if is_first_sent:
                target_text = copy.deepcopy(text_a)
            else:
                target_text = copy.deepcopy(text_b)
            with torch.no_grad():
                for idx in range(len(target_text)):
                    text_tmp = copy.deepcopy(target_text)
                    del text_tmp[idx]
                    if is_first_sent:
                        batch = make_instance(text_tmp, text_b, label, tokenizer, config, label2idx)
                    else:
                        batch = make_instance(text_a, text_tmp, label, tokenizer, config, label2idx)
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, label_ids, token_types = batch[0], batch[1], batch[3], batch[2]

                    outputs = model(input_ids, attention_mask=input_mask, token_type_ids=token_types, labels=label_ids)
                    tmp_eval_loss, logits = outputs[:2]
                    logit = F.softmax(logits, dim=1)[0, label_ids[0]].cpu().data.tolist()
                    logit_lst.append(logit)
                indices = np.argsort(-np.array(logit_lst), axis=0)
                if not len(indices) == len(target_text):
                    ValueError("length of indices doesn't match length of text")
                for i in indices:
                    fd.write(str(i) + '\t')
                fd.write('\n')

def infersent_adversarial_scoring(config, infersent, model, device, eval_dataloader, label_list):
    model.eval()
    label2idx = {label: i for i, label in enumerate(label_list)}
    idx2label = {i: label for i, label in enumerate(label_list)}
    is_single = (config.data_sign == 'SST-2')
    is_first_sent = (config.data_sign == 'SST-2' or config.data_sign == 'MRPC')

    with open(config.pos_file, 'w') as fd:
        for step, batch in tqdm(enumerate(eval_dataloader), desc="Adversarial scoring"):
            with torch.no_grad():
                if is_single:
                    sent1s, _, label_ids = [list(item) for item in batch]
                else:
                    sent1s, sent2s, label_ids = [list(item) for item in batch]
                label_idx_ids = [label2idx[label] for label in label_ids]
                logit_lst = []
                if is_first_sent:
                    target_text_lst = sent1s[0].strip().split(' ')
                else:
                    target_text_lst = sent2s[0].strip().split(' ')
                for idx in range(len(target_text_lst)):
                    sent_tmp = copy.deepcopy(target_text_lst)
                    del sent_tmp[idx]
                    sent_tmps = [' '.join(sent_tmp)]
                    if is_first_sent:
                        sent1_tensor = infersent.encode(sent_tmps, tokenize=True)
                        if not is_single:
                            sent2_tensor = infersent.encode(sent2s, tokenize=True)
                    else:
                        sent1_tensor = infersent.encode(sent1s, tokenize=True)
                        sent2_tensor = infersent.encode(sent_tmps, tokenize=True)


                    if is_single:
                        sent1_tensor = torch.tensor(sent1_tensor)
                        sent1_tensor = sent1_tensor.to(device)
                        logits = model(sent1_tensor)
                    else:
                        sent1_tensor = torch.tensor(sent1_tensor)
                        sent2_tensor = torch.tensor(sent2_tensor)
                        sent1_tensor = sent1_tensor.to(device)
                        sent2_tensor = sent2_tensor.to(device)
                        logits = model(sent1_tensor, sent2_tensor)
                    logit = F.softmax(logits, dim=1)[0, label_idx_ids[0]].data.cpu().tolist()
                    logit_lst.append(logit)
                indices = np.argsort(-np.array(logit_lst), axis=0)
                if not len(indices) == len(target_text_lst):
                    ValueError("length of indices doesn't match length of text")
                for i in indices:
                    fd.write(str(i) + '\t')
                fd.write('\n')

class attack_agent(object):
    def __init__(self, config, error_matrix, device, label_list):
        self.config = config
        self.rate = config.attack_rate
        self.error_matrix = error_matrix
        self.device = device
        self.label_list = label_list
        self.input_file = config.pos_file
        # if the task is a sentence pair task
        self.is_single = (config.data_sign == 'SST-2')
        # if the target sentence is the first sentence in a sentence pair
        self.is_first_sent = (config.data_sign == 'SST-2' or config.data_sign == 'MRPC')

    def attack(self, ):
        NotImplemented


    def wordnet_synonyms(self, word, p, max_num=5):
        synonyms = []
        if p == 'as':
            p = 'a'
            for syn in wordnet.synsets(word, pos=p):
                for l in syn.lemmas():
                    synonyms.append(l.name())
            p = 's'
        for syn in wordnet.synsets(word, pos=p):
            for l in syn.lemmas():
                synonyms.append(l.name())

        # Remove certain kinds of word
        for syn in synonyms:
            if syn == word:
                synonyms.remove(syn)
        for idx, syn in enumerate(synonyms):
            if '-' in syn:
                synonyms.remove(syn)
        for idx, syn in enumerate(synonyms):
            if '_' in syn:
                if len(syn.split('_')) >= 3:
                    synonyms.remove(syn)
                else:
                    synonyms[idx] = syn.replace('_', ' ')

        synonyms = synonyms[:max_num]
        synonyms = list(set(synonyms))
        return synonyms

    def op_sent(self, sent_lst1, op, pos):
        if op == 'WORDER':
            sent_lst1_at = sent_lst1[:pos - 1] + [sent_lst1[pos]] + [sent_lst1[pos - 1]] + sent_lst1[pos + 1:]
        else:
            sent_lst1_at = sent_lst1[:pos] + [op] + sent_lst1[pos + 1:]
        return sent_lst1_at

    def operation_pool(self, pos, sent_tag, error_matrix, max_num=10):
        """
        A simplified version of constructing operation pools.
        We manually ignore some confusion sets since they will have no effect.
        :param pos: the position we will try out.
        :param sent_tag: the pos tags for the whole sentence.
        :param error_matrix: the confusion sets.
        :param max_num: truncate some confusion sets.
        :return: the operation pool along with the error type of each operation.
        """
        ops = []
        ops_type = []
        try:
            tag = sent_tag[pos][1]
        except KeyError:
            print(sent_tag)
            print(pos)

        word = sent_tag[pos][0]
        if len(word) <= 1:
            return ops, ops_type

        if 'VB' in tag:
            ops += lexeme(word)
            if_negate = lambda x: "n't" in x or "not" in x
            for idx, op in enumerate(ops):
                if not if_negate(word) == if_negate(op):
                    del ops[idx]
            l1 = len(ops)
            ops += self.wordnet_synonyms(word, 'v')
            ops_type = ['Vform' if i < l1 else 'Wchoice' for i in range(len(ops))]
            if_sva = lambda word: conjugate(word, '1sg') == word or conjugate(word, '2sg') == word or conjugate(word,
                                                                                                                '3sg') == word
            ops_type = ['SVA' if if_sva(word) else ops_type[idx] for idx, word in enumerate(ops)]

        elif 'NN' in tag:
            ops += self.wordnet_synonyms(word, 'n')
            l1 = len(ops)
            sin_word = inflection.singularize(word)
            plu_word = inflection.pluralize(word)
            if not sin_word == word:
                ops.append(sin_word)
            if not plu_word == word:
                ops.append(plu_word)
            ops_type = ['Wchoice' if i < l1 else 'Nn' for i in range(len(ops))]

        elif 'JJ' in tag:
            ops += self.wordnet_synonyms(word, 'as')
            ops_type = ['Wchoice' for i in range(len(ops))]

        elif 'IN' == tag:
            if word in error_matrix.confusion_matrix['PREP']:
                prob_list = error_matrix.prepps[word]
                indices = np.argsort(-np.array(prob_list))
                indices = indices[:max_num]
                for ind in indices:
                    ops.append(error_matrix.confusion_matrix['PREP'][ind])
            else:
                indices = np.argsort(-np.array(error_matrix.prepsum[word]))
                indices = indices[:max_num]
                for ind in indices:
                    ops.append(error_matrix.confusion_matrix['PREP'][ind])
            ops_type = ['Prep' for i in range(len(ops))]

        elif 'DT' == tag:
            ops = error_matrix.confusion_matrix['ART']
            ops_type = ['ArtOrDet' for i in range(len(ops))]

        elif 'CC' == tag:
            if word in error_matrix.confusion_matrix['TRAN']:
                prob_list = error_matrix.transps[word]
                indices = np.argsort(-np.array(prob_list))
                indices = indices[:max_num]
                for ind in indices:
                    ops.append(error_matrix.confusion_matrix['TRAN'][ind])
            else:
                indices = np.argsort(-np.array(error_matrix.transsum[word]))
                indices = indices[:max_num]
                for ind in indices:
                    ops.append(error_matrix.confusion_matrix['TRAN'][ind])
            ops_type = ['Trans' for i in range(len(ops))]

        elif 'WP' == tag:
            ops = error_matrix.confusion_matrix['WH']
            ops_type = ['Trans' for i in range(len(ops))]

        elif 'WP$' == tag:
            ops = error_matrix.confusion_matrix['WH$']
            ops_type = ['Trans' for i in range(len(ops))]

        elif 'RB' in tag:
            ops += self.wordnet_synonyms(word, 'r')
            l1 = len(ops)
            if pos > 0 and (
                    sent_tag[pos - 1][1] == 'MD' or 'VB' in sent_tag[pos - 1][1] or 'JJ' in sent_tag[pos - 1][1]):
                ops += ['WORDER']
            ops_type = ['Wchoice' if i < l1 else 'Worder' for i in range(len(ops))]

        else:
            pass

        for idx, ele in enumerate(ops):
            if word == ele:
                del ops[idx]
                del ops_type[idx]

        return ops, ops_type

class bert(object):
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

    def make_instance(self, text_a, text_b, label, label_map):
        max_length = self.config.max_length
        if type(text_a) == list:
            text_a = ' '.join(text_a)
        if type(text_b) == list:
            text_b = ' '.join(text_b)
        inputs = self.tokenizer.encode_plus(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                            max_length)

        label = label_map[label]
        input_ids, attention_mask, token_type_ids, label = torch.tensor([input_ids],
                                                                        dtype=torch.long), torch.tensor(
            [attention_mask], dtype=torch.long), torch.tensor([token_type_ids], dtype=torch.long), torch.tensor(
            [label],
            dtype=torch.long)
        return input_ids, attention_mask, token_type_ids, label

    def model_forward(self, model, batch):
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, label_ids, token_types = batch[0], batch[1], batch[3], batch[2]
        outputs = model(input_ids, attention_mask=input_mask, labels=label_ids, token_type_ids=token_types)
        tmp_eval_loss, logits = outputs[:2]
        return logits


class infersent_enc(object):
    def __init__(self, infersent, config):
        self.infersent = infersent
        self.config = config

    def make_instance(self, text_a, text_b, label, label_map):
        sent1s = [' '.join(text_a)]
        if isinstance(text_b, list):
            sent2s = [' '.join(text_b)]
        else:
            sent2s = [text_b]
        return [sent1s, sent2s, [label]]

    def model_forward(self, model, batch):
        sent1s, sent2s, label_ids = [list(item) for item in batch]
        sent1_tensor = self.infersent.encode(sent1s, tokenize=True)
        sent2_tensor = self.infersent.encode(sent2s, tokenize=True)
        sent1_tensor = torch.tensor(sent1_tensor)
        sent2_tensor = torch.tensor(sent2_tensor)
        sent1_tensor = sent1_tensor.to(self.device)
        sent2_tensor = sent2_tensor.to(self.device)
        if not self.config.data_sign == "SST-2":
            logits = model(sent1_tensor, sent2_tensor)
        else:
            logits = model(sent1_tensor)
        return logits



class greedy_attack_agent(attack_agent):
    def __init__(self, config, error_matrix, device, label_list):
        super().__init__(config, error_matrix, device, label_list)


    def attack(self, model, eval_examples, eval_loader):
        model.eval()
        label2idx = {label: i for i, label in enumerate(self.label_list)}
        idx2label = {i: label for i, label in enumerate(self.label_list)}

        with open(self.input_file, 'r') as fi:
            positions = fi.readlines()
        if not (len(positions) == len(eval_examples) or len(positions) == len(eval_loader)):
            raise ValueError("Number of examples not equal")

        with open(self.config.output_att_file, 'w', encoding='utf-8') as fo:
            cnt_sum = []
            cnt_cor = 0
            cnt_att = 0
            for batch_idx, batch in enumerate(eval_loader):
                text_a = eval_examples[batch_idx].text_a.strip().split()
                if self.is_single:
                    text_b = eval_examples[batch_idx].text_b
                else:
                    text_b = eval_examples[batch_idx].text_b.strip().split()
                label = eval_examples[batch_idx].label

                with torch.no_grad():
                    logits = self.model_forward(model, batch)
                    logits = logits.detach().cpu().numpy()
                    logits = np.argmax(logits, axis=-1)
                    if not logits == label2idx[label]:
                        continue

                    cnt_cor += 1

                    pos_sorted = positions[batch_idx].strip().split('\t')
                    pos_sorted = [int(p) for p in pos_sorted]
                    sent_lst1 = text_a
                    sent_lst2 = text_b
                    if self.is_first_sent:
                        sent_tag = nltk.pos_tag(sent_lst1)
                    else:
                        sent_tag = nltk.pos_tag(sent_lst2)
                    sent_len = len(sent_tag)
                    gold_label = label
                    cnt = 1
                    for pos in reversed(pos_sorted):
                        if cnt >= self.rate * sent_len:
                            break

                        op_pool, ops_type = self.operation_pool(pos, sent_tag, self.error_matrix)
                        if len(op_pool) == 0:
                            continue
                        logit_lst = []
                        flag, pos_idx = 0, 0
                        for idx, op in enumerate(op_pool):
                            if self.is_first_sent:
                                sent_lst1_at = self.op_sent(sent_lst1, op, pos)
                                batch = self.make_instance(sent_lst1_at, sent_lst2, label, label2idx)
                            else:
                                sent_lst2_at = self.op_sent(sent_lst2, op, pos)
                                batch = self.make_instance(sent_lst1, sent_lst2_at, label, label2idx)
                            # batch = tuple(t.to(self.device) for t in batch)
                            # input_ids, input_mask, label_ids, token_types = batch[0], batch[1], batch[3], batch[2]
                            # outputs = model(input_ids, attention_mask=input_mask, labels=label_ids, token_type_ids=token_types)
                            # tmp_eval_loss, logits = outputs[:2]
                            logits = self.model_forward(model, batch)
                            logits_bk = logits.detach().cpu().numpy()
                            pred = np.argmax(logits_bk, axis=-1)[0]
                            if not pred == label2idx[gold_label]:
                                flag = 1
                                if self.is_first_sent:
                                    attacked_sent = copy.deepcopy(sent_lst1_at)
                                else:
                                    attacked_sent = copy.deepcopy(sent_lst2_at)
                                pos_idx = idx
                                break
                            else:
                                logit = F.softmax(logits, dim=1)[0, label2idx[gold_label]].data.cpu().tolist()
                                logit_lst.append(logit)
                        if flag:
                            cnt_sum.append(cnt / len(attacked_sent))
                            cnt_att += 1
                            fo.write(' '.join(attacked_sent))
                            fo.write('\n')
                            break
                        indices = np.argsort(np.array(logit_lst), axis=0)
                        if self.is_first_sent:
                            sent_lst1 = self.op_sent(sent_lst1, op_pool[indices[0]], pos)
                        else:
                            sent_lst2 = self.op_sent(sent_lst2, op_pool[indices[0]], pos)
                        cnt += 1
            return 100.0 * sum(cnt_sum) / len(cnt_sum), 100.0 * cnt_att / cnt_cor

class beam_search_attack_agent(attack_agent):
    def __init__(self, config, error_matrix, device, label_list):
        super().__init__(config, error_matrix, device, label_list)
        self.beam_size = config.beam_size


    def attack(self, model, eval_examples, eval_loader):
        model.eval()
        label2idx = {label: i for i, label in enumerate(self.label_list)}
        idx2label = {i: label for i, label in enumerate(self.label_list)}

        with open(self.input_file, 'r') as fi:
            positions = fi.readlines()
        assert len(positions) == len(eval_loader)
        assert len(positions) == len(eval_examples)

        with open(self.config.output_att_file, 'w', encoding='utf-8') as fo:
            cnt_sum = []
            cnt_cor = 0
            cnt_att = 0
            for batch_idx, batch in enumerate(eval_loader):
                text_a = eval_examples[batch_idx].text_a.strip().split()
                if self.is_single:
                    text_b = eval_examples[batch_idx].text_b
                else:
                    text_b = eval_examples[batch_idx].text_b.strip().split()
                label = eval_examples[batch_idx].label
                with torch.no_grad():
                    logits = self.model_forward(model, batch)
                    logits = logits.detach().cpu().numpy()
                    logits = np.argmax(logits, axis=-1)
                    if not logits == label2idx[eval_examples[batch_idx].label]:
                        continue

                    cnt_cor += 1

                    pos_sorted = positions[batch_idx].strip().split('\t')
                    pos_sorted = [int(p) for p in pos_sorted]
                    sent_lst1 = text_a
                    sent_lst2 = text_b
                    if self.is_first_sent:
                        sent_tag = nltk.pos_tag(sent_lst1)
                    else:
                        sent_tag = nltk.pos_tag(sent_lst2)
                    sent_len = len(sent_tag)
                    gold_label = label

                    if self.is_first_sent:
                        sent_beam = [copy.deepcopy(sent_lst1) for _ in range(self.beam_size)]
                    else:
                        sent_beam = [copy.deepcopy(sent_lst2) for _ in range(self.beam_size)]

                    cnt = 1
                    for pos in reversed(pos_sorted):

                        if cnt >= self.rate * sent_len:
                            break

                        op_pool, op_type = self.operation_pool(pos, sent_tag, self.error_matrix)
                        if len(op_pool) == 0:
                            continue

                        logit_lst = []
                        flag = 0

                        for sent_idx, sent_lst in enumerate(sent_beam):
                            for op_idx, op in enumerate(op_pool):
                                if self.is_first_sent:
                                    sent_lst1_at = self.op_sent(sent_lst, op, pos)
                                    batch = self.make_instance(sent_lst1_at, sent_lst2, label, label2idx)
                                else:
                                    sent_lst2_at = self.op_sent(sent_lst, op, pos)
                                    batch = self.make_instance(sent_lst1, sent_lst2_at, label, label2idx)

                                logits = self.model_forward(model, batch)
                                logits_bk = logits.detach().cpu().numpy()
                                pred = np.argmax(logits_bk, axis=-1)[0]

                                if not pred == label2idx[gold_label]:
                                    flag = 1
                                    if self.is_first_sent:
                                        attacked_sent = copy.deepcopy(sent_lst1_at)
                                    else:
                                        attacked_sent = copy.deepcopy(sent_lst2_at)
                                    break
                                else:
                                    logit = F.softmax(logits, dim=1)[0, label2idx[gold_label]].cpu().data.tolist()
                                    logit_lst.append((op_idx, sent_idx, logit))
                            if flag or cnt == 1:
                                break
                        if flag:
                            cnt_sum.append(cnt / len(attacked_sent))
                            fo.write(' '.join(attacked_sent))
                            fo.write('\n')
                            cnt_att += 1
                            break
                        sorted_logits = sorted(logit_lst, key=lambda x: x[2])

                        if not cnt == 1:
                            for idx, sent in enumerate(sent_beam):
                                sent_beam[idx] = self.op_sent(sent_beam[sorted_logits[idx][1]],
                                                         op_pool[sorted_logits[idx][0]], pos)
                        else:
                            for idx, sent in enumerate(sent_beam):
                                if idx > len(sorted_logits) - 1:
                                    # in case beam size bigger than operation pool
                                    sent_beam[idx] = self.op_sent(sent, op_pool[sorted_logits[-1][0]], pos)
                                else:
                                    sent_beam[idx] = self.op_sent(sent, op_pool[sorted_logits[idx][0]], pos)

                        cnt += 1

            return 100.0 * sum(cnt_sum) / len(cnt_sum), 100.0 * cnt_att / cnt_cor

class genetic_attack_agent(attack_agent):
    def __init__(self, config, error_matrix, device, label_list):
        super().__init__(config, error_matrix, device, label_list)
        self.beam_size = config.beam_size
        self.max_iter_rate = config.max_iter_rate
        self.pop_size = config.pop_size

    def select_best_replacement(self, idx, sent_cur, gold_label, replace_list, model, sent_lst1,
                                label2idx, sent_lst2):
        new_x_list = [self.op_sent(sent_cur, w, idx) for w in replace_list]
        logit_lst = []
        with torch.no_grad():
            for sent_p in new_x_list:
                batch = self.make_instance(sent_p, sent_lst2, gold_label, label2idx)
                logits = self.model_forward(model, batch)
                logits_bk = logits.cpu().detach().numpy()

                logit = F.softmax(logits, dim=1)[0, label2idx[gold_label]].cpu().data.tolist()
                logit_lst.append(logit)
        indices = np.argsort(np.array(logit_lst), axis=0)
        return new_x_list[indices[0]]

    def generate_population(self, neighbours_lst, w_select_probs, gold_label, model, sent_lst1, sent_lst2, label2idx):
        return [self.perturb(sent_lst1, neighbours_lst, w_select_probs, model, sent_lst1, gold_label, label2idx, sent_lst2) for _ in range(self.pop_size)]

    def crossover(self, sent1, sent2):
        sent_new = copy.deepcopy(sent1)
        for i in range(len(sent_new)):
            if np.random.uniform() < 0.5:
                sent_new[i] = sent2[i]
        return sent_new

    def perturb(self, sent_cur, neighbours_list, w_select_probs, model, sent_lst1, gold_label, label2idx, sent_lst2):
        x_len = w_select_probs.shape[0]
        random_idx = np.random.choice(x_len, size=1, p=w_select_probs)[0]

        while sent_cur[random_idx] != sent_lst1[random_idx] and np.sum(sent_lst1 != sent_cur) < np.sum(
                                                                        np.sign(w_select_probs)):
            random_idx = np.random.choice(x_len, size=1, p=w_select_probs)[0]
        replace_list = neighbours_list[random_idx]
        return self.select_best_replacement(random_idx, sent_cur, gold_label, replace_list, model,
                                            sent_lst1, label2idx, sent_lst2)

    def attack(self, model, eval_examples, eval_loader):

        model.eval()
        label2idx = {label: i for i, label in enumerate(self.label_list)}
        idx2label = {i: label for i, label in enumerate(self.label_list)}
        # tokenizer = self.tokenizer

        with open(self.config.output_att_file, 'w', encoding='utf-8') as fo:
            cnt_sum = []
            cnt = 0
            cnt_cor = 0
            cnt_att = 0
            for batch_idx, batch in enumerate(eval_loader):

                text_a = eval_examples[batch_idx].text_a.strip().split()
                if self.is_single:
                    text_b = eval_examples[batch_idx].text_b
                else:
                    text_b = eval_examples[batch_idx].text_b.strip().split()
                label = eval_examples[batch_idx].label
                with torch.no_grad():
                    logits = self.model_forward(model, batch)
                    logits = logits.cpu().detach().numpy()
                    logits = np.argmax(logits, axis=-1)
                    if not logits == label2idx[eval_examples[batch_idx].label]:
                        continue

                    cnt_cor += 1

                    sent_lst1 = text_a
                    sent_lst2 = text_b
                    sent_tag = nltk.pos_tag(sent_lst1)
                    sent_len = len(sent_tag)
                    gold_label = label

                    neighbours_lst = [self.operation_pool(i, sent_tag, self.error_matrix)[0] for i in range(sent_len)]
                    neighbours_len = [len(i) for i in neighbours_lst]
                    w_select_probs = neighbours_len / np.sum(neighbours_len)
                    pop = self.generate_population(neighbours_lst, w_select_probs, gold_label, model,
                                              sent_lst1, sent_lst2, label2idx)
                    max_iter = 0
                    for i in w_select_probs:
                        if not i == 0:
                            max_iter += 1
                    flag = 0
                    max_iter = int(max_iter * self.max_iter_rate)
                    for i in range(max_iter):
                        logit_lst = []
                        for sent_p in pop:
                            batch = self.make_instance(sent_p, sent_lst2, label, label2idx)
                            logits = self.model_forward(model, batch)
                            logits_bk = logits.cpu().detach().numpy()
                            pred = np.argmax(logits_bk, axis=-1)[0]
                            if not pred == label2idx[gold_label]:
                                flag = 1
                                cnt_sum.append((i + 1) / len(sent_p))
                                attacked_sent = copy.deepcopy(sent_p)
                                break
                            else:
                                logit = F.softmax(logits, dim=1)[0, label2idx[gold_label]].cpu().data.tolist()
                                logit_lst.append(logit)
                        if flag:
                            break
                        indices = np.argsort(np.array(logit_lst), axis=0)
                        elite = [pop[indices[-1]]]
                        # print(logits)
                        select_probs = np.array(logit_lst)
                        select_probs /= select_probs.sum()
                        # select_probs = F.softmax(torch.Tensor(logits), dim=0)
                        p1 = np.random.choice(self.pop_size, size=self.pop_size - 1, p=select_probs)
                        p2 = np.random.choice(self.pop_size, size=self.pop_size - 1, p=select_probs)
                        childs = [self.crossover(pop[p1[idx]], pop[p2[idx]]) for idx in range(self.pop_size - 1)]
                        childs = [self.perturb(x, neighbours_lst, w_select_probs, model, sent_lst1,
                                          gold_label, label2idx, sent_lst2) for x in childs]
                        pop = elite + childs
                    if flag:
                        cnt += 1
                        cnt_att += 1
                        print("total: %d" % cnt_cor)
                        print('attacked %d' % cnt)
                        fo.write(' '.join(attacked_sent))
                        fo.write('\n')
                        continue

            return 100.0 * sum(cnt_sum) / len(cnt_sum), 100.0 * cnt_att / cnt_cor



class bert_greedy_attack_agent(greedy_attack_agent, bert):
    def __init__(self, config, error_matrix, tokenizer, device, label_list):
        greedy_attack_agent.__init__(self, config, error_matrix, device, label_list)
        bert.__init__(self, tokenizer, config)

class bert_beam_search_attack_agent(beam_search_attack_agent, bert):
    def __init__(self, config, error_matrix, tokenizer, device, label_list):
        beam_search_attack_agent.__init__(self, config, error_matrix, device, label_list)
        bert.__init__(self, tokenizer, config)

class bert_genetic_attack_agent(genetic_attack_agent, bert):
    def __init__(self, config, error_matrix, tokenizer, device, label_list):
        genetic_attack_agent.__init__(self, config, error_matrix, device, label_list)
        bert.__init__(self, tokenizer, config)

class infersent_greedy_attack_agent(greedy_attack_agent, infersent_enc):
    def __init__(self, config, error_matrix, infersent, device, label_list):
        greedy_attack_agent.__init__(self, config, error_matrix, device, label_list)
        infersent_enc.__init__(self, infersent, config)

class infersent_beam_search_attack_agent(beam_search_attack_agent, infersent_enc):
    def __init__(self, config, error_matrix, infersent, device, label_list):
        beam_search_attack_agent.__init__(self, config, error_matrix, device, label_list)
        infersent_enc.__init__(self, infersent, config)

class infersent_genetic_attack_agent(genetic_attack_agent, infersent_enc):
    def __init__(self, config, error_matrix, infersent, device, label_list):
        genetic_attack_agent.__init__(self, config, error_matrix, device, label_list)
        infersent_enc.__init__(self, infersent, config)



