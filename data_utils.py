import os
import sys


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import csv
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


class TextDataset(Dataset):
    def __init__(self, *texts):
        assert all(len(texts[0]) == len(text) for text in texts)
        self.texts = texts

    def __getitem__(self, index):
        return tuple(text[index] for text in self.texts)

    def __len__(self):
        return len(self.texts[0])


class DataProcessor(object):
    # processor for the query-based ner dataset
    def get_train_examples(self, data_dir):
        data = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        return data#[:len(data)//32]

    def get_dev_examples(self, data_dir):
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_test_examples(self, data_dir):
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                # if len(line) < 5:
                #     print(line)
                #     continue
                lines.append(line)
            return lines


class MRPCProcessor(DataProcessor):
    def get_labels(self, ):
        return ["0", "1"]

    def get_train_examples(self, data_dir):
        """See base class."""
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        print(data_dir)
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, path=None):
        """See base class."""
        if path is None:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "test")
        else:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, path)), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            try:
                text_b = line[4]
            except:
                continue
            if text_b.strip() == '':
                continue
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QNLIProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, path=None):
        """See base class."""
        if path is None:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "test")
        else:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, path)), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class SSTProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, path=None):
        """See base class."""
        if path is None:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "test")
        else:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, path)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir, path=None):
        """See base class."""
        if path is None:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")
        else:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, path)), "test")
            
    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            if i < 6:
                print(text_a)
                print(text_b)
                print(label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid 
        self.text_a = text_a 
        self.text_b = text_b 
        self.label = label 

