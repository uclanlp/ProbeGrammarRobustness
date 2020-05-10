# train and evaluate classifier for each probing task
import os
import sys
root_path = '/'.join(os.path.realpath(__file__).split('/')[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)
import argparse
import json
import random
import torch
import torch.nn as nn
import numpy as np
from tools import MLP, self_attn_mlp
from transformers import BertTokenizer
from transformers import BertModel, BertConfig
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def classify(args, train_X, train_y, dev_X, dev_y, test_X, test_y, model, tokenizer):
    classifier_config = {'nhid': args.nhid, 'optim': 'adam', 'batch_size': args.batch_size, 'tenacity': 2, 'max_epoch': 10, 'epoch_size': 4, 'dropout': args.dropout}
    reg = 10**(-3)
    props, scores = [], []
    feat_dim = 768 #if args.bert_model.startswith('bert-base') else 1024
    num_classes = 2

    clf = self_attn_mlp(classifier_config, inputdim=feat_dim, nclasses=num_classes, l2reg=reg, seed=args.seed, cudaEfficient=True)
    clf.fit(args, model, tokenizer, train_X, train_y, dev_X, dev_y)
    scores.append(round(100 * clf.score(args, model, tokenizer, dev_X), 2))
    props.append([reg])

    opt_prop = props[np.argmax(scores)]
    dev_acc = np.max(scores)
    print('model saved')
    test_acc = round(100 * clf.score(args, model, tokenizer, test_X), 2)
    print("best reg = %.2f; dev score = %.4f; test score = %.4f;"%(opt_prop[0], dev_acc, test_acc))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    torch.save(clf.model.state_dict(), os.path.join(args.output_dir, 'params_{1}_{2}nucle_{0}.pkl'.format(str(args.layer), args.type.strip(), args.bert_model)))


total = 0
def init_weights(m):
    global total
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        total += m.weight.size(0)*m.weight.size(1)
        total += m.bias.size(0)
    elif type(m) == torch.nn.Embedding:
        torch.nn.init.xavier_uniform(m.weight)
        total += m.weight.size(0)*m.weight.size(1)
    elif hasattr(m, 'weight') and hasattr(m, 'bias'):
        total += m.weight.size(0)
        total += m.bias.size(0)

def load_model(args):

    if args.transformer_model.startswith('bert'):
        path = '/home/yinfan/.cache/torch/transformers/bert-base-uncased-pytorch_model.bin'
        config = BertConfig.from_pretrained(args.transformer_model, output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained(args.transformer_model, do_lower_case=True)
        model = BertModel.from_pretrained(path, from_tf=bool('.ckpt' in args.transformer_model),
                                              config=config)
    else:
        path = '/home/yinfan/.cache/torch/transformers/roberta-base-pytorch_model.bin'
        tokenizer = RobertaTokenizer.from_pretrained(args.transformer_model)
        config = RobertaConfig.from_pretrained(args.transformer_model, output_hidden_states=True)
        model = RobertaModel.from_pretrained(path, from_tf=bool('.ckpt' in args.transformer_model),
                                                  config=config)
    # roberta = RobertaModel.from_pretrained(args.roberta_model, cache_dir=args.cache_dir, config=config)
    model_embedding = model.embeddings
    model_embedding.to(args.device)
    if args.n_gpu > 1:
        model_embedding = torch.nn.DataParallel(model_embedding)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.untrained_transformer == 1:
        model.apply(init_weights)
    return model, model_embedding, tokenizer

def load_data(input_file):
    """Read a list of `InputExample`s from an input file."""
    train_X, train_y, dev_X, dev_y, test_X, test_y = [], [], [], [], [], []
    cat2id, id2cat = {}, {}
    unique_id_tr = 0
    unique_id_va = 0
    unique_id_te = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            try:
                line = reader.readline()
            except:
                continue
            if not line:
                break
            try:
                split, lab, text, _ = line.split('\t')
            except:
                split, lab, text = line.split('\t')

            if lab not in cat2id:
                cat2id[lab] = len(id2cat)
                id2cat[cat2id[lab]] = lab
            y = cat2id[lab]
            if split == 'tr':
                train_X.append(InputExample(unique_id=unique_id_tr, text=text))
                train_y.append(y)
                unique_id_tr += 1
            elif split == 'va':
                dev_X.append(InputExample(unique_id=unique_id_va, text=text))
                dev_y.append(y)
                unique_id_va += 1
            elif split == 'te':
                test_X.append(InputExample(unique_id=unique_id_te, text=text))
                test_y.append(y)
                unique_id_te += 1
    return train_X, train_y, dev_X, dev_y, test_X, test_y

class InputExample(object):
    def __init__(self, unique_id, text):
        self.unique_id = unique_id
        self.text = text

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, label_id):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.label_id = label_id

def convert_examples_to_features(examples, labels, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, pack) in enumerate(zip(examples,labels)):
        example, label = pack[0], pack[1]
        cand_tokens = tokenizer.tokenize(example.text)
        # Account for [CLS] and [SEP] with "- 2"
        if len(cand_tokens) > seq_length - 2:
            cand_tokens = cand_tokens[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in cand_tokens:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                label_id=label))
    return features

def get_max_seq_length(instances, tokenizer):
    max_seq_len = -1
    for instance in instances:
        cand_tokens = tokenizer.tokenize(instance.text)
        cur_len = len(cand_tokens)
        if cur_len > max_seq_len:
            max_seq_len = cur_len
    return max_seq_len

def main():
    parser = argparse.ArgumentParser(description="Probing classifier")
    parser.add_argument("--data_file",
                      type=str,
                      default=None,
                      help="file containing probing text and labels")
    parser.add_argument("--output_dir",
                      type=str,
                      default='./save_bert_self_attn',
                      help="output directory for trained self-attention classifier")
    parser.add_argument('--layer',
                      type=int,
                      default=0,
                      help='bert layer id to probe')
    parser.add_argument('--nhid',
                      type=int,
                      default=50,
                      help='hidden size of MLP')
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--cache_dir", default="/local/fanyin/tmp", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='batch size per step')
    parser.add_argument('--dropout',
                      type=float,
                      default=0.0,
                      help='dropout prob. value')
    parser.add_argument('--seed',
                      type=int,
                      default=123,
                      help='seed value to be set manually')
    parser.add_argument("--transformer-model",
                        default="bert-base-uncased",
                        type=str,
                        help="bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased")
    parser.add_argument("--untrained_transformer",
                        type=int,
                        help="use untrained version of bert")
    parser.add_argument("--type", default='prep', type=str)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu= torch.cuda.device_count()
    args.device = device
    set_seed(args)
    print(args)
    model, model_embedding, tokenizer = load_model(args)
    print('loading_data')
    train_X, train_y, dev_X, dev_y, test_X, test_y = load_data(args.data_file)
    train_features = convert_examples_to_features(
        examples=train_X, labels=train_y, seq_length=2 + get_max_seq_length(train_X, tokenizer), tokenizer=tokenizer)
    dev_features = convert_examples_to_features(
        examples=dev_X, labels=dev_y, seq_length=2 + get_max_seq_length(dev_X, tokenizer), tokenizer=tokenizer)
    test_features = convert_examples_to_features(
        examples=test_X, labels=test_y, seq_length=2 + get_max_seq_length(test_X, tokenizer), tokenizer=tokenizer)
    print('start classifying!')
    classify(args, train_features, train_y, dev_features, dev_y, test_features, test_y, model, tokenizer)

if __name__ == "__main__":
  main()
