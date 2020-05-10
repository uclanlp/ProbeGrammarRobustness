#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from models import InferSent, NLINet, ClassificationNet
from data_utils import *
import os
import sys
import time
import argparse
import copy

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import nltk
from utils.Error_all import Errors
from utils.statistic_all import stat, dis
from attack_agent import *
from pattern3.en import conjugate, lemma, lexeme
from nltk.corpus import wordnet
import inflection
import random

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_infersent():
    V = 2
    MODEL_PATH = 'encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))
    W2V_PATH = 'fastText/crawl-300d-2M.vec'
    infersent.set_w2v_path(W2V_PATH)
    infersent.build_vocab_k_words(K=100000)
    return infersent


def args_parser():
    # start parser 
    parser = argparse.ArgumentParser()

    # requires parameters
    parser.add_argument("--target_model", default='infersent', type=str)
    parser.add_argument("--mode", default='fine-tune', help='options: fine-tune, score, attack', type=str)
    parser.add_argument("--data_dir", default="./", type=str)
    parser.add_argument("--pos_dir", default="./pos", type=str)
    parser.add_argument("--attack_dir", default="./attacked_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="/home/yinfan/robustGrammar/fanyin_data/saved_models")

    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--dev_batch_size", default=256, type=int)
    parser.add_argument("--test_batch_size", default=256, type=int)

    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--checkpoint", default=50, type=int)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--export_model", type=bool, default=True)

    parser.add_argument("--data_sign", type=str, default="MRPC")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--enc_lstm_dim", type=int, default=2048)
    parser.add_argument("--fc_dim", type=int, default=512)

    parser.add_argument("--adversarial", action='store_true')
    parser.add_argument("--attack_rate", type=float, default=0.15)
    parser.add_argument("--adv_type", type=str, default='greedy')
    parser.add_argument("--random_attack_file", type=str, default=None)
    parser.add_argument("--beam_size", type=int, default=5)
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


def load_data(config):
    print("-*-" * 10)
    print("current data_sign: {}".format(config.data_sign))

    if config.data_sign == "MRPC":
        data_processor = MRPCProcessor()
    elif config.data_sign == "QNLI":
        data_processor = QNLIProcessor()
    elif config.data_sign == "MNLI":
        data_processor = MnliProcessor()
    elif config.data_sign == "SST-2":
        data_processor = SSTProcessor()
    else:
        raise ValueError("Please Notice that your data_sign DO NOT exits !!!!!")

    label_list = data_processor.get_labels()
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
    for idx, example in enumerate(train_examples):
        if example.text_b is None:
            train_examples[idx].text_b = '<p>'
    for idx, example in enumerate(dev_examples):
        if example.text_b is None:
            dev_examples[idx].text_b = '<p>'
    for idx, example in enumerate(test_examples):
        if example.text_b is None:
            test_examples[idx].text_b = '<p>'
    train_data = TextDataset([example.text_a for example in train_examples], [example.text_b
                                                                              for example in train_examples], [example.label for example in train_examples])
    train_sampler = SequentialSampler(train_data)
    dev_data = TextDataset([example.text_a for example in dev_examples], [example.text_b
                                                                          for example in dev_examples], [example.label for example in dev_examples])
    dev_sampler = SequentialSampler(dev_data)
    test_data = TextDataset([example.text_a for example in test_examples], [example.text_b
                                                                            for example in test_examples], [example.label for example in test_examples])
    test_sampler = SequentialSampler(test_data)
    print("check loaded data")

    train_dataloader = DataLoader(train_data, sampler=train_sampler, \
                                  batch_size=config.train_batch_size)

    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, \
                                batch_size=config.dev_batch_size)

    test_dataloader = DataLoader(test_data, sampler=test_sampler, \
                                 batch_size=config.test_batch_size)

    num_train_steps = int(
        len(train_examples) / config.train_batch_size / config.gradient_accumulation_steps * config.num_train_epochs)
    return train_examples, dev_examples, test_examples, train_dataloader, dev_dataloader, test_dataloader, num_train_steps, label_list


def load_model(config):
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    infersent = load_infersent()
    infersent.to(device)
    if not config.data_sign == 'SST-2':
        model = NLINet(config)
    else:
        model = ClassificationNet(config)
    model.to(device)
    if config.mode == 'score' or config.mode == 'attack':
        model_dict_path = os.path.join(config.output_dir,
                                         "{}_{}.bin".format(config.data_sign, config.target_model))
        model.load_state_dict(torch.load(model_dict_path))
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return infersent, model, optimizer, device, n_gpu


def train(infersent, model, optimizer, train_dataloader, dev_dataloader, test_dataloader, config, \
          device, n_gpu, label_list):
    model.train()
    global_step = 0
    label2idx = {label: i for i, label in enumerate(label_list)}
    idx2label = {i: label for i, label in enumerate(label_list)}
    is_single = config.data_sign == "SST-2"
    loss_fc = CrossEntropyLoss().cuda()
    
    dev_best_acc = 0

    test_best_acc = 0

    try:
        for idx in range(int(config.num_train_epochs)):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            print("#######" * 10)
            print("EPOCH: ", str(idx))
            last_step_eval = False
            for step, batch in tqdm(enumerate(train_dataloader)):
                if is_single:
                    sent1s, _, label_ids = [list(item) for item in batch]
                else:
                    sent1s, sent2s, label_ids = [list(item) for item in batch]

                label_idx_ids = [label2idx[label] for label in label_ids]
                label_idx_ids = torch.tensor(label_idx_ids, dtype=torch.long).to(device)
                with torch.no_grad():
                    if is_single:
                        sent1_tensor = infersent.encode(sent1s, tokenize=True)
                    else:
                        sent1_tensor = infersent.encode(sent1s, tokenize=True)
                        sent2_tensor = infersent.encode(sent2s, tokenize=True)
                if is_single:
                    sent1_tensor = torch.tensor(sent1_tensor)
                    sent1_tensor = sent1_tensor.to(device)
                    output = model(sent1_tensor)
                else:
                    sent1_tensor = torch.tensor(sent1_tensor)
                    sent2_tensor = torch.tensor(sent2_tensor)
                    sent1_tensor = sent1_tensor.to(device)
                    sent2_tensor = sent2_tensor.to(device)
                    output = model(sent1_tensor, sent2_tensor)

                loss = loss_fc(output, label_idx_ids)
                if n_gpu > 1:
                    loss = loss.mean()
                if config.gradient_accumulation_steps > 1:
                    loss = loss / config.gradient_accumulation_steps
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += sent1_tensor.size(0)
                nb_tr_steps += 1

                if (step + 1) % config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                if nb_tr_steps % (config.checkpoint * config.gradient_accumulation_steps) == 0 and not last_step_eval:
                    print("-*-" * 15)
                    print("current training loss is : ")
                    print(loss.item())
                    tmp_dev_acc = eval_checkpoint(model, infersent, dev_dataloader, config, device, n_gpu, label_list, eval_sign="dev")
                    print("......" * 10)
                    print("DEV: acc")
                    print(tmp_dev_acc)

                    if tmp_dev_acc > dev_best_acc:
                        dev_best_acc = tmp_dev_acc
                        tmp_test_acc = eval_checkpoint(model, infersent, test_dataloader, config, device, n_gpu,
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
                                output_model_file = os.path.join(config.output_dir, "{}_{}.bin".format(config.data_sign, config.target_model))
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

def eval_checkpoint(model_object, infersent, eval_dataloader, config, \
                    device, n_gpu, label_list, eval_sign="dev"):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader 
    model_object.eval()
    label2idx = {label: i for i, label in enumerate(label_list)}
    idx2label = {i: label for i, label in enumerate(label_list)}
    is_single = config.data_sign == 'SST-2'
    eval_loss = 0
    pred_lst = []
    gold_lst = []
    eval_steps = 0

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            if is_single:
                sent1s, _, label_ids = [list(item) for item in batch]
                sent1_tensor = infersent.encode(sent1s, tokenize=True)
            else:
                sent1s, sent2s, label_ids = [list(item) for item in batch]
                sent1_tensor = infersent.encode(sent1s, tokenize=True)
                sent2_tensor = infersent.encode(sent2s, tokenize=True)
            label_idx_ids = [label2idx[label] for label in label_ids]

            if is_single:
                sent1_tensor = torch.tensor(sent1_tensor)
                sent1_tensor = sent1_tensor.to(device)
                logits = model_object(sent1_tensor)
            else:
                sent1_tensor = torch.tensor(sent1_tensor)
                sent2_tensor = torch.tensor(sent2_tensor)
                sent1_tensor = sent1_tensor.to(device)
                sent2_tensor = sent2_tensor.to(device)
                logits = model_object(sent1_tensor, sent2_tensor)
            logits = logits.cpu().detach().numpy()
            preds = np.argmax(logits, axis=-1)
            pred_lst += list(preds)
            gold_lst += label_idx_ids
    cnt = 0
    for pred, gold in zip(pred_lst, gold_lst):
        if pred == gold:
           cnt += 1
    return 1.0 * cnt / len(pred_lst)


def random_attack(config, infersent, model, device, n_gpu, dev_loader, test_loader, label_list):
    model.eval()
    label2idx = {label: i for i, label in enumerate(label_list)}
    idx2label = {i: label for i, label in enumerate(label_list)}

    pred_lst = []
    gold_lst = []
    error_lst = []
    is_single = config.data_sign == 'SST-2'

    for step, batch in enumerate(dev_loader):
        with torch.no_grad():
            if is_single:
                sent1s, _, label_ids = [list(item) for item in batch]
                sent1_tensor = infersent.encode(sent1s, tokenize=True)
            else:
                sent1s, sent2s, label_ids = [list(item) for item in batch]
                sent1_tensor = infersent.encode(sent1s, tokenize=True)
                sent2_tensor = infersent.encode(sent2s, tokenize=True)
            label_idx_ids = [label2idx[label] for label in label_ids]

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
            logits = logits.cpu().detach().numpy()
            preds = np.argmax(logits, axis=-1)
            pred_lst += list(preds)
            gold_lst += label_idx_ids
    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            if is_single:
                sent1s, _, label_ids = [list(item) for item in batch]
                sent1_tensor = infersent.encode(sent1s, tokenize=True)
            else:
                sent1s, sent2s, label_ids = [list(item) for item in batch]
                sent1_tensor = infersent.encode(sent1s, tokenize=True)
                sent2_tensor = infersent.encode(sent2s, tokenize=True)

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
    print('acc drop: ', 1.0 * cnt / all)


def adversarial_attack(config, infersent, model, device, n_gpu, dev_examples, dev_loader, test_loader, label_list, type='greedy'):
    preps, dets, trans = stat()
    error_matrix = Errors(preps, dets, trans)
    if type == 'greedy':
        agent = infersent_greedy_attack_agent(config, error_matrix, infersent, device, label_list)
    elif type == 'beam_search':
        agent = infersent_beam_search_attack_agent(config, error_matrix, infersent, device, label_list)
    elif type == 'genetic':
        agent = infersent_genetic_attack_agent(config, error_matrix, infersent, device, label_list)
    elif type == 'random':
        random_attack(config, infersent, model, device, n_gpu, dev_loader, test_loader, label_list)
        return
    logger.info('start attacking')
    per_rate, att_rate = agent.attack(model, dev_examples, dev_loader)
    logger.info('{} attack finished: attack success rate {:.2f}%, changed {:.2f}% tokens'.format(config.adv_type,
                                                                                                 att_rate, per_rate))

def main():
    config = args_parser()
    if config.mode == 'fine-tune':
        train_examples, dev_examples, test_examples, train_loader, dev_loader, test_loader, num_train_steps, label_list = load_data(config)
        config.n_classes = len(label_list)
        infersent, model, optimizer, device, n_gpu = load_model(config)
        train(infersent, model, optimizer, train_loader, dev_loader, test_loader, config, device, n_gpu, label_list)
    else:
        config.train_batch_size = 1
        config.dev_batch_size = 1
        config.test_batch_size = 1
        train_examples, dev_examples, test_examples, train_loader, dev_loader, test_loader, num_train_steps, label_list = load_data(config)
        config.n_classes = len(label_list)
        infersent, model, optimizer, device, n_gpu = load_model(config)
        model.eval()

        tmp_dev_acc = eval_checkpoint(model, infersent, dev_loader, config, device, n_gpu, label_list, eval_sign="dev")

        logger.info('checked loaded model, current dev score: {}'.format(tmp_dev_acc))

        if not os.path.exists(config.pos_dir):
            os.mkdir(config.pos_dir)
        config.pos_file = os.path.join(config.pos_dir, '{}_{}_pos.txt'.format(config.data_sign, config.target_model))
        if not os.path.exists(config.attack_dir):
            os.mkdir(config.attack_dir)
        config.output_att_file = os.path.join(config.attack_dir,
                                              '{}_{}_{}.txt'.format(config.data_sign, config.target_model, config.adv_type))
        if config.mode == 'score':
            infersent_adversarial_scoring(config, infersent, model, device, dev_loader, label_list)
        else:
            adversarial_attack(config, infersent, model, device, n_gpu, dev_examples, dev_loader, test_loader,
                               label_list, type=config.adv_type)


if __name__ == "__main__":
    main()
