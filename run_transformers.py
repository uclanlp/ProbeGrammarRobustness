# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from data_utils import *

import os
import sys
import time
import argparse
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

from utils.Error_all import Errors
from utils.statistic_all import stat, dis
from attack_agent import *

import nltk
from pattern3.en import conjugate, lemma, lexeme
from nltk.corpus import wordnet
import inflection

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def args_parser():
    # start parser
    parser = argparse.ArgumentParser()

    # requires parameters
    parser.add_argument("--target_model", default='roberta', help='target model, options: Bert, RoBERTa', type=str)
    parser.add_argument("--mode", default='fine-tune', help='options: fine-tune, score, attack', type=str)
    parser.add_argument("--model_name_or_path", default='bert-base-cased', type=str)
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--data_sign", type=str, default="MRPC")

    parser.add_argument("--data_dir", default="./", type=str)
    parser.add_argument("--pos_dir", default="./pos", type=str)
    parser.add_argument("--attack_dir", default="./attacked_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="/home/yinfan/robustGrammar/fanyin_data/saved_bert")

    parser.add_argument("--train_batch_size", default=24, type=int)
    parser.add_argument("--dev_batch_size", default=256, type=int)
    parser.add_argument("--test_batch_size", default=256, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--checkpoint", default=50, type=int)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--export_model", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_proportion", default=0.00, type=float)
    parser.add_argument("--threads", default=12, type=int)

    parser.add_argument("--adversarial", action='store_true')
    parser.add_argument("--attack_rate", type=float, default=0.15)
    parser.add_argument("--adv_type", type=str, default='greedy')
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--random_attack_file", type=str, default=None)
    parser.add_argument("--pop_size", type=int, default=60)
    parser.add_argument("--max_iter_rate", type=float, default=0.23)

    args = parser.parse_args()

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    return args

class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def convert_examples_to_features(examples,
                                      tokenizer=None,
                                      max_length=512,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    label_map = {label: i for i, label in enumerate(label_list)}
    output_mode = 'classification'
    features = []
    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples to features"):

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label))
    return features


def load_data(config):
    print("-*-" * 10)
    print("current data_sign: {}".format(config.data_sign))
    config.data_sign = config.data_sign.strip()
    if config.data_sign == "MRPC":
        data_processor = MRPCProcessor()
    elif config.data_sign == "QNLI":
        data_processor = QNLIProcessor()
    elif config.data_sign == "MNLI":
        data_processor = MnliProcessor()
    elif config.data_sign == "SST-2":
        data_processor = SSTProcessor()
    else:
        raise ValueError("Data sign doesn't exist")

    label_list = data_processor.get_labels()
    if config.target_model == "bert":
        tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path, do_lower_case=config.do_lower_case)
    else:
        if config.data_sign == "MNLI":
            label_list[1], label_list[2] = label_list[2], label_list[1]
        tokenizer = RobertaTokenizer.from_pretrained(config.model_name_or_path, do_lower_case=config.do_lower_case)
    print(label_list)

    # load data exampels
    train_examples = data_processor.get_train_examples(config.data_dir)
    dev_examples = data_processor.get_dev_examples(config.data_dir)
    if config.random_attack_file is not None:
        test_examples = data_processor.get_test_examples(config.data_dir, config.random_attack_file)
    else:
        test_examples = data_processor.get_test_examples(config.data_dir)
    print(len(train_examples))
    print(len(dev_examples))
    print(len(test_examples))
    print("check loaded data")


    train_features = convert_examples_to_features(train_examples, tokenizer, label_list=label_list, max_length=config.max_length)
    dev_features = convert_examples_to_features(dev_examples, tokenizer, label_list=label_list, max_length=config.max_length)
    test_features = convert_examples_to_features(test_examples, tokenizer, label_list=label_list, max_length=config.max_length)

    train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    train_input_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
    train_segment_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)
    train_label_ids = torch.tensor([f.label for f in train_features], dtype=torch.long)
    train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label_ids)

    dev_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
    dev_input_mask = torch.tensor([f.attention_mask for f in dev_features], dtype=torch.long)
    dev_segment_ids = torch.tensor([f.token_type_ids for f in dev_features], dtype=torch.long)
    dev_label_ids = torch.tensor([f.label for f in dev_features], dtype=torch.long)
    dev_data = TensorDataset(dev_input_ids, dev_input_mask, dev_segment_ids, dev_label_ids)

    test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    test_input_mask = torch.tensor([f.attention_mask for f in test_features], dtype=torch.long)
    test_segment_ids = torch.tensor([f.token_type_ids for f in test_features], dtype=torch.long)
    test_label_ids = torch.tensor([f.label for f in test_features], dtype=torch.long)
    test_data = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_label_ids)
    train_sampler = SequentialSampler(train_data)
    dev_sampler = SequentialSampler(dev_data)
    test_sampler = SequentialSampler(test_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, \
                                  batch_size=config.train_batch_size)

    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, \
                                batch_size=config.dev_batch_size)

    test_dataloader = DataLoader(test_data, sampler=test_sampler, \
                                 batch_size=config.test_batch_size)

    num_train_steps = int(
        len(train_examples) / config.train_batch_size / config.gradient_accumulation_steps * config.num_train_epochs)
    return train_examples, dev_examples, test_examples, train_dataloader, dev_dataloader, test_dataloader, num_train_steps, label_list


def load_model(args, num_train_steps, label_list):

    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    if not args.target_model in args.model_name_or_path:
        raise ValueError("Target model and model path not match, please check your setting")

    if args.target_model == 'bert':
        path = '/home/yinfan/.cache/torch/transformers/bert-base-uncased-pytorch_model.bin'
        config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=len(label_list))
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        # model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
        #                                     config=config)
        model = BertForSequenceClassification.from_pretrained(path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                              config=config)
    else:
        path = '/home/yinfan/.cache/torch/transformers/roberta-base-pytorch_model.bin'
        config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=len(label_list))
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = RobertaForSequenceClassification.from_pretrained(path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config)

    if args.mode == 'score' or args.mode == 'attack':
        model_dict_path = os.path.join(args.output_dir,
                                         "{}_{}.bin".format(args.data_sign, args.target_model))
        model.load_state_dict(torch.load(model_dict_path))
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * num_train_steps), num_training_steps=num_train_steps
    )
    return model, tokenizer, optimizer, scheduler, device, n_gpu


def train(model, optimizer, scheduler, train_dataloader, dev_dataloader, test_dataloader, config, \
          device, n_gpu, label_list):

    global_step = 0
    label2idx = {label: i for i, label in enumerate(label_list)}
    idx2label = {i: label for i, label in enumerate(label_list)}

    dev_best_acc = 0

    test_best_acc = 0

    try:
        for idx in range(int(config.num_train_epochs)):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            print("#######" * 10)
            print("EPOCH: ", str(idx))
            last_step_eval = False
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, label_ids, token_types = batch[0], batch[1], batch[3], batch[2]
                outputs = model(input_ids, attention_mask=input_mask,token_type_ids=token_types,labels=label_ids)
                loss = outputs[0]
                if n_gpu > 1:
                    loss = loss.mean()
                if config.gradient_accumulation_steps > 1:
                    loss = loss / config.gradient_accumulation_steps
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % config.gradient_accumulation_steps == 0:

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                if nb_tr_steps % (config.checkpoint * config.gradient_accumulation_steps) == 0 and not last_step_eval:
                    print("-*-" * 15)
                    print("current training loss is : ")
                    print(loss.item())

                    tmp_dev_acc = eval_checkpoint(model, dev_dataloader, config, device, n_gpu, label_list,
                                                  eval_sign="dev")
                    print("......" * 10)
                    print("DEV: acc")
                    print(tmp_dev_acc)

                    if tmp_dev_acc > dev_best_acc:
                        dev_best_acc = tmp_dev_acc
                        tmp_test_acc = eval_checkpoint(model, test_dataloader, config, device, n_gpu,
                                                       label_list, eval_sign="test")
                        print("......" * 10)
                        print("TEST: acc")
                        print(tmp_test_acc)
                        print("......" * 10)

                        if tmp_test_acc > test_best_acc:
                            test_best_acc = tmp_test_acc

                            # export model
                            if config.export_model:
                                model_to_save = model.module if hasattr(model, "module") else model
                                output_model_file = os.path.join(config.output_dir,
                                                                 "{}_{}.bin".format(config.data_sign, config.target_model))
                                torch.save(model_to_save.state_dict(), output_model_file)

                    print("-*-" * 15)
                    last_step_eval = True
                else:
                    last_step_eval = False

    except KeyboardInterrupt:
        print("=&=" * 15)
        print("DEV: current best acc")
        print(dev_best_acc)
        print("TEST: current best acc")
        print(test_best_acc)
        print("=&=" * 15)

    print("=&=" * 15)
    print("DEV: current best acc")
    print(dev_best_acc)
    print("TEST: current best acc")
    print(test_best_acc)
    print("=&=" * 15)



def eval_checkpoint(model_object, eval_dataloader, config, \
                    device, n_gpu, label_list, eval_sign="dev"):

    model_object.eval()

    pred_lst = []
    gold_lst = []

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids, token_types = batch[0], batch[1], batch[3], batch[2]
            outputs = model_object(input_ids, attention_mask=input_mask, token_type_ids=token_types, labels=label_ids)
            tmp_eval_loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            logits = np.argmax(logits, axis=-1)

            pred_lst += list(logits)
            gold_lst += label_ids.detach().cpu().tolist()

    cnt = 0
    for pred, gold in zip(pred_lst, gold_lst):
        if pred == gold:
            cnt += 1
    return 1.0 * cnt / len(pred_lst)


def random_attack(config, model, device, n_gpu, dev_loader, test_loader, label_list):
    model.eval()

    pred_lst = []
    gold_lst = []
    error_lst = []

    for step, batch in enumerate(dev_loader):
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids, token_types = batch[0], batch[1], batch[3], batch[2]
            outputs = model(input_ids, attention_mask=input_mask, token_type_ids=token_types, labels=label_ids)
            tmp_eval_loss, logits = outputs[:2]
            logits = logits.cpu().detach().numpy()
            preds = np.argmax(logits, axis=-1)
            pred_lst += list(preds)
            gold_lst += label_ids.cpu().detach().tolist()

    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids, token_types = batch[0], batch[1], batch[3], batch[2]
            outputs = model(input_ids, attention_mask=input_mask, token_type_ids=token_types, labels=label_ids)
            tmp_eval_loss, logits = outputs[:2]
            logits = logits.cpu().detach().numpy()
            preds = np.argmax(logits, axis=-1)

            error_lst += list(preds)

    cnt = 0
    all = 0
    for pred, err, gold in zip(pred_lst, error_lst, gold_lst):
        if not pred == gold:
            continue
        all += 1
        if not pred == err:
            cnt += 1
    print(cnt)
    print('acc drop: ', 1.0 * cnt / all)
    cnt = 0
    for pred, gold in zip(pred_lst, gold_lst):
        if pred == gold:
            cnt += 1
    print('ori_acc: ', 1.0 * cnt / len(pred_lst))


def adversarial_attack(config, model, tokenizer, device, n_gpu, dev_examples, test_examples, dev_loader, test_loader, label_list,
                       type='greedy'):
    preps, dets, trans = stat()
    error_matrix = Errors(preps, dets, trans)
    if type == 'greedy':
        agent = bert_greedy_attack_agent(config, error_matrix, tokenizer, device, label_list)
    elif type == 'beam_search':
        agent = bert_beam_search_attack_agent(config, error_matrix, tokenizer, device, label_list)
    elif type == 'genetic':
        agent = bert_genetic_attack_agent(config, error_matrix, tokenizer, device, label_list)
    elif type == 'random':
        random_attack(config, model, device, n_gpu, dev_loader, test_loader, label_list)
        return
    logger.info('start attacking')
    per_rate, att_rate = agent.attack(model, dev_examples, dev_loader)
    logger.info('{} attack finished: attack success rate {:.2f}%, changed {:.2f}% tokens'.format(config.adv_type, att_rate, per_rate))

def main():
    config = args_parser()
    if config.mode == 'fine-tune':
        train_examples, dev_examples, test_examples, train_loader, dev_loader, test_loader, num_train_steps, label_list = load_data(
            config)
        model, tokenizer, optimizer, scheduler, device, n_gpu = load_model(config, num_train_steps, label_list)
        config.n_classes = len(label_list)
        train(model, optimizer, scheduler, train_loader, dev_loader, test_loader, config, device, n_gpu, label_list)
    else:
        config.train_batch_size = 1
        config.dev_batch_size = 1
        config.test_batch_size = 1
        train_examples, dev_examples, test_examples, train_loader, dev_loader, test_loader, num_train_steps, label_list = load_data(
            config)
        model, tokenizer, optimizer, scheduler, device, n_gpu = load_model(config, num_train_steps, label_list)
        model.eval()
        config.n_classes = len(label_list)
        tmp_dev_acc = eval_checkpoint(model, dev_loader, config, device, n_gpu,
                                      label_list, eval_sign="dev")
        logger.info('checked loaded model, current dev score: {}'.format(tmp_dev_acc))
        if not os.path.exists(config.pos_dir):
            os.mkdir(config.pos_dir)
        config.pos_file = os.path.join(config.pos_dir, '{}_{}_pos.txt'.format(config.data_sign, config.target_model))
        if not os.path.exists(config.attack_dir):
            os.mkdir(config.attack_dir)
        config.output_att_file = os.path.join(config.attack_dir,
                                              '{}_{}_{}.txt'.format(config.data_sign, config.target_model, config.adv_type))
        if config.mode == 'score':
            adversarial_scoring(config, tokenizer, model, device, dev_examples, label_list)
        else:
            adversarial_attack(config, model, tokenizer, device, n_gpu, dev_examples, test_examples, dev_loader,
                               test_loader, label_list, type=config.adv_type)




if __name__ == "__main__":
    main()

