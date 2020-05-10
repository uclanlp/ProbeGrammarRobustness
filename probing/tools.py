# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Codes are borrowed from https://github.com/facebookresearch/SentEval/blob/master/senteval/tools/classifier.py
 and https://github.com/ganeshjawahar/interpret_bert
 with small modifications and our own implement of a simple self-attention layer

Pytorch Classifier class in the style of scikit-learn
Classifiers include Logistic Regression and MLP
"""

from __future__ import absolute_import, division, unicode_literals
import os
import sys
root_path = '/'.join(os.path.realpath(__file__).split('/')[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)
import numpy as np
import copy
from senteval import utils

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
import torch.nn.functional as F
from tqdm import tqdm

class PyTorchClassifier(object):
  def __init__(self, inputdim, nclasses, l2reg=0., batch_size=64, seed=1111,
               cudaEfficient=False):
    # fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    self.inputdim = inputdim
    self.nclasses = nclasses
    self.l2reg = l2reg
    self.batch_size = batch_size
    self.cudaEfficient = cudaEfficient

  def prepare_data(self, args, features):

      all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
      all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
      all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
      all_segment_ids = torch.tensor([f.input_type_ids for f in features], dtype=torch.long)
      all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
      eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index, all_segment_ids, all_label_ids)
      eval_sampler = RandomSampler(eval_data)
      eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)
      return eval_dataloader, eval_sampler

  def fit(self, args, model, tokenizer, train_x, train_y, dev_x, dev_y, validation_split=None,
          early_stop=True):
    self.nepoch = 0
    bestaccuracy = -1
    stop_train = False
    early_stop_count = 0

    # Preparing validation data
    train_dataloader, train_sampler = self.prepare_data(args, train_x)
    # Training
    while not stop_train and self.nepoch <= self.max_epoch:
      self.trainepoch(args, model, tokenizer, train_dataloader, epoch_size=self.epoch_size)
      accuracy = self.score(args, model, tokenizer, dev_x)
      if accuracy > bestaccuracy:
        bestaccuracy = accuracy
        bestmodel = copy.deepcopy(self.model)
      elif early_stop:
        if early_stop_count >= self.tenacity:
          stop_train = True
        early_stop_count += 1
      self.model = bestmodel
    return bestaccuracy

  def trainepoch(self, args, model, tokenizer, train_dataloader, epoch_size=1, log_step = 50):
    all_costs = []
    for _ in range(self.nepoch, self.nepoch + epoch_size):
      for step, batch in enumerate(train_dataloader):
          batch = tuple(t.to(args.device) for t in batch)
          inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[3]}
          ybatch = batch[4]
          with torch.no_grad():
            _, _, all_encoder_layers = model(**inputs)
          layer_output = all_encoder_layers[args.layer]
          self.model.train()
          output = self.model(layer_output, batch[1].type(torch.cuda.FloatTensor))
          loss = self.loss_fn(output, ybatch)
          all_costs.append(loss.data.item())
          # backward
          self.optimizer.zero_grad()
          loss.backward()
          # Update parameters
          self.optimizer.step()
    self.nepoch += epoch_size

  def score(self, args, model, tokenizer, dev_x):
    dev_dataloader, dev_sampler = self.prepare_data(args, dev_x)
    self.model.eval()
    correct = 0
    all = 0
    with torch.no_grad():
      for step, batch in enumerate(dev_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[3]}
        ybatch = batch[4]
        with torch.no_grad():
          _, _, all_encoder_layers = model(**inputs)
        layer_output = all_encoder_layers[args.layer]
        output = self.model(layer_output, batch[1].type(torch.cuda.FloatTensor))
        output_pred = output.cpu().data.tolist()
        pred = []
        for p in output_pred:
          pred.append(0 if p[0] > p[1] else 1)
        yb = ybatch.data.tolist()
        for p, g in zip(pred, yb):
          all += 1
          if p == g:
            correct += 1
    accuracy = 1.0 * correct / all
    return accuracy

  def predict(self, devX):
    self.model.eval()
    devX = torch.FloatTensor(devX).cuda()
    yhat = np.array([])
    with torch.no_grad():
      for i in range(0, len(devX), self.batch_size):
        Xbatch = devX[i:i + self.batch_size]
        output = self.model(Xbatch)
        yhat = np.append(yhat, output.data.max(1)[1].cpu().numpy())
    yhat = np.vstack(yhat)
    return yhat

  def predict_proba(self, devX):
    self.model.eval()
    probas = []
    with torch.no_grad():
      for i in range(0, len(devX), self.batch_size):
        Xbatch = devX[i:i + self.batch_size]
        vals = F.softmax(self.model(Xbatch).data.cpu().numpy())
        if not probas:
          probas = vals
        else:
          probas = np.concatenate(probas, vals, axis=0)
    return probas

"""
MLP with Pytorch (nhid=0 --> Logistic Regression)
"""

class MLP(PyTorchClassifier):
  def __init__(self, params, inputdim, nclasses, l2reg=0., batch_size=64,
               seed=1111, cudaEfficient=False):
    super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                         batch_size, seed, cudaEfficient)
    """
    PARAMETERS:
    -nhid:       number of hidden units (0: Logistic Regression)
    -optim:      optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
    -tenacity:   how many times dev acc does not increase before stopping
    -epoch_size: each epoch corresponds to epoch_size pass on the train set
    -max_epoch:  max number of epoches
    -dropout:    dropout for MLP
    """

    self.nhid = 0 if "nhid" not in params else params["nhid"]
    self.optim = "adam" if "optim" not in params else params["optim"]
    self.tenacity = 5 if "tenacity" not in params else params["tenacity"]
    self.epoch_size = 4 if "epoch_size" not in params else params["epoch_size"]
    self.max_epoch = 200 if "max_epoch" not in params else params["max_epoch"]
    self.dropout = 0. if "dropout" not in params else params["dropout"]
    self.batch_size = 64 if "batch_size" not in params else params["batch_size"]

    if params["nhid"] == 0:
      self.model = nn.Sequential(
        nn.Linear(self.inputdim, self.nclasses),
    ).cuda()
    else:
      self.model = nn.Sequential(
        nn.Linear(self.inputdim, params["nhid"]),
        nn.Dropout(p=self.dropout),
        nn.Sigmoid(),
        nn.Linear(params["nhid"], self.nclasses),
      ).cuda()

    self.loss_fn = nn.CrossEntropyLoss().cuda()
    self.loss_fn.size_average = False

    optim_fn, optim_params = utils.get_optimizer(self.optim)
    self.optimizer = optim_fn(self.model.parameters(), **optim_params)
    self.optimizer.param_groups[0]['weight_decay'] = self.l2reg

class self_attn_mlp(PyTorchClassifier):
  def __init__(self, params, inputdim, nclasses, l2reg=0., batch_size=64,
               seed=1111, cudaEfficient=False):
    super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                         batch_size, seed, cudaEfficient)
    """
    PARAMETERS:
    -nhid:       number of hidden units (0: Logistic Regression)
    -optim:      optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
    -tenacity:   how many times dev acc does not increase before stopping
    -epoch_size: each epoch corresponds to epoch_size pass on the train set
    -max_epoch:  max number of epoches
    -dropout:    dropout for MLP
    """

    self.nhid = 0 if "nhid" not in params else params["nhid"]
    self.optim = "adam" if "optim" not in params else params["optim"]
    self.tenacity = 5 if "tenacity" not in params else params["tenacity"]
    self.epoch_size = 10 if "epoch_size" not in params else params["epoch_size"]
    self.max_epoch = 200 if "max_epoch" not in params else params["max_epoch"]
    self.dropout = 0. if "dropout" not in params else params["dropout"]
    self.batch_size = 64 if "batch_size" not in params else params["batch_size"]
    self.model = self_attn(self.inputdim, self.dropout, self.nhid, nclasses).cuda()

    self.loss_fn = nn.CrossEntropyLoss().cuda()
    self.loss_fn.size_average = False

    optim_fn, optim_params = utils.get_optimizer(self.optim)
    self.optimizer = optim_fn(self.model.parameters(), **optim_params)
    self.optimizer.param_groups[0]['weight_decay'] = self.l2reg

class self_attn(nn.Module):
  def __init__(self, input_dim, dropout, nhid, nlabels):
    super(self.__class__, self).__init__()
    self._input_dim = input_dim
    self.Ws1 = nn.Linear(input_dim, input_dim // 2)
    self.Ws2 = nn.Linear(input_dim // 2, 1)
    self.dropout = dropout
    self.tanh = nn.Tanh()
    self.attn_dropout = nn.Dropout(p=self.dropout)
    self.attn_softmax = nn.Softmax(dim=1)
    self.cls = nn.Linear(input_dim, nlabels)

  def forward(self, Xbatch, mask):
    value = self.attn_dropout(self.Ws1(Xbatch))
    attn_score = self.Ws2(self.tanh(value))
    mask = (1.0 - mask.unsqueeze(2)) * -10000.0
    attn_score = attn_score + mask
    attn_score = self.attn_softmax(attn_score)
    attn_output = torch.bmm(attn_score.transpose(1,2), Xbatch).squeeze(1)
    output = self.cls(attn_output)
    return output
