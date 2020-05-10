# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Evaluate the masked language model function of BERT on minimal pairs

import os
import sys
root_path = '/'.join(os.path.realpath(__file__).split('/')[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)
import argparse
import json
import random
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler

from transformers import BertConfig, BertTokenizer
from transformers import BertModel, BertForMaskedLM
from aligner import align_bert
import torch.nn.functional as F
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def prepare_data(args, features):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_lm_masked = torch.tensor([f.lm_label_ids for f in features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index, all_segment_ids, all_lm_masked)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)
    return eval_dataloader, eval_sampler


def evaluate(args, features, bert):
    bert.eval()
    dataloader, data_sampler = prepare_data(args, features)
    probs = 0.0
    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[3],
                  'masked_lm_labels': batch[4]}

        with torch.no_grad():
            output_mlm = bert(**inputs)

            mlm_loss, pred_score = output_mlm[:2]
            label_mask = (inputs["masked_lm_labels"] == -100)
            index = inputs["masked_lm_labels"].masked_fill(label_mask, value=0).unsqueeze(-1).repeat(1, 1, pred_score.size(-1))
            pred_score = F.softmax(pred_score, dim=2)
            # print(pred_score)
            scores = torch.gather(pred_score, -1, index)
            scores = scores[:, :, 0].masked_fill(label_mask, value=0)
            probs += torch.sum(scores).item() / torch.sum((scores != 0)).item()
    return probs / len(dataloader)


def load_bert(args):
    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True, cache_dir=args.cache_dir)
    # bert = BertForMaskedLM.from_pretrained(args.bert_model, cache_dir=args.cache_dir)
    path = '/home/yinfan/.cache/torch/transformers'
    bert_config = BertConfig.from_pretrained(args.model_name_or_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    bert = BertForMaskedLM.from_pretrained(path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                          config=bert_config)
    bert.to(args.device)
    if args.n_gpu > 1:
        bert = torch.nn.DataParallel(bert)
    return bert, tokenizer


def load_data(config):
    """Read a list of `InputExample`s from an input file."""
    cor_file = os.path.join(config.data_dir, "{}_cor.txt".format(config.type.strip()))
    err_file = os.path.join(config.data_dir, "{}_err.txt".format(config.type.strip()))
    corrects = []
    errors = []
    positions = []
    with open(cor_file, 'r') as fc, open(err_file, 'r') as fe:
        clines = fc.readlines()
        elines = fe.readlines()
    cnt = 0
    for cline, eline in zip(clines, elines):
        cnt += 1
        try:
            correct, pos1 = cline.strip().split('\t')
        except:
            correct = cline.strip()
            pos1 = '-1'
        try:
            error, pos2 = eline.strip().split('\t')
        except:
            error = eline.strip()
            pos2 = '-1'

        assert pos1 == pos2

        corrects.append(correct)
        errors.append(error)
        positions.append(int(pos2))
    return corrects, errors, positions


class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self, tokens, input_ids, input_mask, segment_ids, lm_label_ids):
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.lm_label_ids = lm_label_ids



def convert_examples_to_features(args, examples, seq_length, tokenizer, positions):
    """Loads a data file into a list of `InputBatch`s."""
    cor_features = []
    err_features = []
    cor_toks = []
    err_toks = []
    cor_examples = examples[0]
    err_examples = examples[1]
    cor_positions = positions
    err_positions = positions
    for (ex_index, pack) in enumerate(zip(cor_examples, err_examples, cor_positions, err_positions)):
        cor_example, err_example, cor_posi, err_posi = pack[0], pack[1], pack[2], pack[3]
        cor_posi = cor_posi + args.offset
        err_posi = err_posi + args.offset
        cand_tokens_cor = tokenizer.tokenize(cor_example)
        cand_tokens_err = tokenizer.tokenize(err_example)
        # Account for [CLS] and [SEP] with "- 2"
        if len(cand_tokens_cor) > seq_length - 2:
            cand_tokens_cor = cand_tokens_cor[0:(seq_length - 2)]
        if len(cand_tokens_err) > seq_length - 2:
            cand_tokens_err = cand_tokens_err[0:(seq_length - 2)]
        cor_tokens = []
        cor_input_type_ids = []
        cor_tokens.append("[CLS]")
        cor_input_type_ids.append(0)
        err_tokens = []
        err_input_type_ids = []
        err_tokens.append("[CLS]")
        err_input_type_ids.append(0)
        for cor_token, err_token in zip(cand_tokens_cor, cand_tokens_err):
            cor_tokens.append(cor_token)
            err_tokens.append(err_token)
            cor_input_type_ids.append(0)
            err_input_type_ids.append(0)
        cor_tokens.append("[SEP]")
        cor_input_type_ids.append(0)
        err_tokens.append("[SEP]")
        err_input_type_ids.append(0)
        cor_ta = align_bert(cor_example, cor_tokens)
        err_ta = align_bert(err_example, err_tokens)

        try:
            cor_pos = cor_ta.project_tokens(cor_posi).tolist()
            err_pos = err_ta.project_tokens(err_posi).tolist()
        except:
            continue

        cor_masked_lm_positions = [i for i in cor_pos]
        cor_masked_lm_labels = [cor_tokens[i] for i in cor_pos]
        for i in cor_pos:
            cor_tokens[i] = '[MASK]'
        cor_input_ids = tokenizer.convert_tokens_to_ids(cor_tokens)
        cor_masked_label_ids = tokenizer.convert_tokens_to_ids(cor_masked_lm_labels)
        cor_input_mask = [1] * len(cor_input_ids)
        cor_lm_label_array = np.full(seq_length, dtype=np.int, fill_value=-100)
        cor_lm_label_array[cor_masked_lm_positions] = cor_masked_label_ids
        err_masked_lm_positions = [i for i in err_pos]
        err_masked_lm_labels = [err_tokens[i] for i in err_pos]
        for i in err_pos:
            err_tokens[i] = '[MASK]'
        err_input_ids = tokenizer.convert_tokens_to_ids(err_tokens)
        err_masked_label_ids = tokenizer.convert_tokens_to_ids(err_masked_lm_labels)
        err_input_mask = [1] * len(err_input_ids)
        err_lm_label_array = np.full(seq_length, dtype=np.int, fill_value=-100)
        err_lm_label_array[err_masked_lm_positions] = err_masked_label_ids 

      # Zero-pad up to the sequence length.
        while len(cor_input_ids) < seq_length:
            cor_input_ids.append(0)
            cor_input_mask.append(0)
            cor_input_type_ids.append(0)
        while len(err_input_ids) < seq_length:
            err_input_ids.append(0)
            err_input_mask.append(0)
            err_input_type_ids.append(0)

        assert len(cor_input_ids) == seq_length
        assert len(cor_input_mask) == seq_length
        assert len(cor_input_type_ids) == seq_length
        cor_features.append(
            InputFeatures(
                tokens=cor_tokens,
                input_ids=cor_input_ids,
                input_mask=cor_input_mask,
                segment_ids=cor_input_type_ids,
                lm_label_ids=cor_lm_label_array))

        err_features.append(
            InputFeatures(
                tokens=err_tokens,
                input_ids=err_input_ids,
                input_mask=err_input_mask,
                segment_ids=err_input_type_ids,
                lm_label_ids=err_lm_label_array))

    return cor_features, err_features, cor_toks, err_toks

def get_max_seq_length(instances, tokenizer):
    max_seq_len = -1
    for instance in instances:
        cand_tokens = tokenizer.tokenize(instance)
        cur_len = len(cand_tokens)
        if cur_len > max_seq_len:
            max_seq_len = cur_len
    return max_seq_len


def main():
    parser = argparse.ArgumentParser(description="Probing MLM")
    parser.add_argument("--data_dir", default="../examples/", type=str)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--model_name_or_path", default='bert-base-cased', type=str)
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--cache_dir", default="/home/yinfan/tmp", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='batch size per step')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.0,
                        help='dropout prob. value')
    parser.add_argument('--seed',
                        type=int,
                        default=123,
                        help='seed value to be set manually')
    parser.add_argument("--bert_model",
                        default="bert-base-uncased",
                        type=str,
                        help="bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased")
    parser.add_argument("--type",
                        type=str,
                        help="target type of error, options: Prep, ArtOrDet, Wci, Trans, Nn, SVA, Vform")
    parser.add_argument("--offset", type=int, default=1, help="offset to the error position")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)
    bert, tokenizer = load_bert(args)
    logger.info("loading data...")
    correct_sentences, error_sentences, positions = load_data(args)

    cor_features, err_features, cor_toks, err_toks = convert_examples_to_features(args, 
        examples=(correct_sentences, error_sentences), seq_length=2 + get_max_seq_length(correct_sentences + error_sentences, tokenizer), tokenizer=tokenizer, positions=positions)

    logger.info("start evaluating...")
    cor_prob = evaluate(args, cor_features, bert)
    err_prob = evaluate(args, err_features, bert)
    logger.info('Masked LM Performance without Errors: {:.3f}%'.format(100 * cor_prob))
    logger.info('Masked LM Performance with Errors: {:.3f}%'.format(100 * err_prob))
    logger.info('Performance Drop: {:.3f}%'.format(100 * (cor_prob - err_prob)))
if __name__ == "__main__":
    main()
