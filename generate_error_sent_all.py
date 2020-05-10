# !/usr/bin/env python
# -*- coding: utf-8 -*-


import random
import os
import numpy as np
import math
from utils.Tree_code_all import Node, Tree
from utils.Error_all import Errors
from io import StringIO
from utils.statistic_all import stat, dis
from functools import partial
import json
import copy
import pandas as pd
import argparse
import sys


IDX = 0

def peek(line):  # See next char w/o moving position
    pos = line.tell()
    char = line.read(1)
    line.seek(pos)
    return char


def parse_token(line):  # Get next token in line
    char = line.read(1)
    while char == " ":
        char = line.read(1)
    if not char:
        return None
    token = char
    if token == "(" or token == ")":
        return token
    while peek(line) != " " and peek(line) != ")" and peek(line):
        token += line.read(1)
    return token

def parse_expression(line):  # recursively build tree of operators & operands
    global IDX
    token = parse_token(line)
    if not token or token == ")":
        return None
    children = []
    if token == "(":
        token = parse_token(line)
        while peek(line) != ")" and peek(line):
            children.append(parse_expression(line))
        if peek(line) == ")":
            line.read(1)
    if not children:
        token = (token, IDX)
        IDX += 1
    return Node(token, children)

def print_tree(node, level = 0):
    if not node:
        return
    print('*' * level)
    print(node.value)
    level += 1
    for chld in node.children:
        print_tree(chld, level=level)

def change_sent(sent, idx_lsts, error_type_lst, error_matrix, preps, dets, trans, pos):
    sent = sent.strip().split(' ')
    # print(sent)
    # 0 for not changed, 1 for substitute, 2 for insert, 3 for delete
    modi = ['0' for tok in sent]
    cnt = 0
    for pos_lst, error_type in zip(idx_lsts, error_type_lst):
        pos_lst = [pos for pos in pos_lst if modi[pos] == '0']
        if error_type == 'false_prep':
            id_type = error_matrix.intro_prep_error(sent, pos_lst, preps, pos)
        elif error_type == 'false_plural':
            id_type = error_matrix.intro_nn_error(sent, pos_lst, 'p')
        elif error_type == 'false_singular':
            id_type = error_matrix.intro_nn_error(sent, pos_lst, 's')
        elif error_type == 'false_art':
            id_type = error_matrix.intro_art_error(sent, pos_lst, dets, pos)
        elif error_type == 'false_vt':
            id_type = error_matrix.intro_vt_error(sent, pos_lst)
        elif error_type == 'false_wform':
            id_type = error_matrix.intro_wform_error(sent, pos_lst)
        elif error_type == 'false_tran':
            id_type = error_matrix.intro_trans_error(sent, pos_lst, trans, pos)
        elif error_type == 'false_woinc':
            pos_lst = [pos for pos in pos_lst if pos < len(modi) - 1 and modi[pos + 1] == '0']
            id_type = error_matrix.intro_worder_error(sent, pos_lst)
        elif error_type == 'false_woadv':
            pos_lst = [pos for pos in pos_lst if pos < len(modi) - 1 and modi[pos + 1] == '0']
            id_type = error_matrix.intro_worder_error(sent, pos_lst)
        elif error_type == 'false_3sg':
            id_type = error_matrix.intro_sva_error(sent, pos_lst, '3sg')
        elif error_type == 'false_n3sg':
            id_type = error_matrix.intro_sva_error(sent, pos_lst, 'n3sg')
        # word order changed
        if isinstance(id_type, list):
            cnt += 1
            modi[id_type[0][0]] = str(id_type[0][1])
            modi[id_type[1][0]] = str(id_type[1][1])
        else:
            if not id_type == (0, 0):
                cnt += 1
                modi[id_type[0]] = str(id_type[1])
    return sent, modi

def find_pos(error_matrix, sent, line, error_num, pos=[]):
    global IDX
    coarse_dis = error_matrix.coarse_grained_dis
    fine_dis = error_matrix.fine_grained_dis
    false_sets = error_matrix.false_sets
    idx_lists = list()
    error_type_lst = list()
    # First sample the num of errors for each general category, then sample errors for each fine-grained category
    # sample coarse-grained types

    coar_type = np.random.choice([0, 1], error_num, p=coarse_dis)
    for type in coar_type:
        error = np.random.choice(false_sets[type], 1, p=fine_dis[type])
        error_type_lst.append(error[0])

    for idx, error in enumerate(error_type_lst):
        error_type = error
        if line.strip().count('(') != line.strip().count(')'):
            error_type_lst = list()
            return idx_lists, error_type_lst
        elif line.strip() != "( (X (SYM )) )" and line.strip():
            IDX = 0
            line_temp = line.strip()[2:-2]
            line_temp = StringIO(line_temp)  # treat line as file
            tree = Tree(parse_expression(line_temp))  # create tree
            idx_list = list()
            try:
                if error_type == 'false_prep':
                    tree.find_prep_ins(tree.root, idx_list)
                if error_type == 'false_plural':
                    tree.find_pl_noun(tree.root, idx_list)
                elif error_type == 'false_singular':
                    tree.find_sng_nouns(tree.root, idx_list)
                if error_type == 'false_art':
                    tree.find_det_ins(tree.root, idx_list)
                if error_type == 'false_vt':
                    tree.find_verb(tree.root, idx_list)
                if error_type == 'false_wform':
                    tree.find_wform(tree.root, idx_list)
                if error_type == 'false_3sg':
                    tree.find_3sg(tree.root, idx_list)
                elif error_type == 'false_n3sg':
                    tree.find_n3sg(tree.root, idx_list)
                if error_type == 'false_tran':
                    idx_list.append(0)
                if error_type == 'false_woadv':
                    tree.find_advp(tree.root, idx_list)
                elif error_type == 'false_woinc':
                    tree.find_worder(tree.root, idx_list)
            except:
                pass
            # Exclude some positions
            idx_list = [i for i in idx_list if (i not in pos) and (i < len(sent.strip().split(' ')))]
            idx_lists.append(idx_list)
        else:
            error_type_lst = list()
            return idx_lists, error_type_lst
    return idx_lists, error_type_lst


def get_text_pos_from_line(pos):
    sent = pos.strip()
    p = list()
    pp = list()
    return sent, p, pp

def change_pos(modi, ori_p):
    aft_p = copy.deepcopy(ori_p)
    for idx, item in enumerate(modi):
        if item == '2':
            aft_p = [p + int(p > idx) for p in aft_p]
        elif item == '3':
            aft_p = [p - int(p > idx) for p in aft_p]
    return aft_p

def change_label(modi, ori_p):
    after_p = list()
    for idx, item in enumerate(ori_p):
        if modi[idx] == '0':
            after_p.append(item)
        elif modi[idx] == '1':
            after_p.append('X')
        elif modi[idx] == '2':
            after_p.append('X')
            after_p.append(item)
    return after_p

def run(arguments):
    preps, dets, trans = stat()
    error_matrix = Errors(preps, dets, trans)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_sign', default='MRPC', help='dataset', type=str)
    parser.add_argument('--input_tsv', help='input tsv input', type=str)
    parser.add_argument('--parsed_sent1', help='parse tree for sent1', type=str)
    parser.add_argument('--parsed_sent2', help='parse tree for sent2', type=str)
    parser.add_argument('--output_tsv', help='output csv file', type=str)
    parser.add_argument('--rate', help='error rate', type=str)
    args = parser.parse_args(arguments)
    input_tsv = args.input_tsv
    parsed_file1 = args.parsed_sent1
    parsed_file2 = args.parsed_sent2
    output_tsv = args.output_tsv
    rate = args.rate

    with open(input_tsv, 'r') as ff, open(parsed_file1, 'r') as fp1, open(parsed_file2, 'r') as fp2:
        read_rows = partial(pd.read_csv,
                            sep='\t',
                            error_bad_lines=False,
                            header=None,
                            skiprows=1,
                            quoting=3,
                            keep_default_na=False,
                            encoding="utf-8", )
        if args.data_sign == 'SST-2':
            rows = read_rows(input_tsv, names=["sentence", "label"])
        elif args.data_sign == 'MRPC':
            rows = read_rows(input_tsv, names=["Quality", "#1 ID", "#2 ID", "#1 String", "#2 String"])
        elif args.data_sign == 'MNLI':
            rows = read_rows(input_tsv, names=['index', 'captionID', 'pairID', 'sentence1_binary_parse',
                                                'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse',
                                                'sentence1', 'sentence2', 'label1', 'label2', 'label3', 'label4',
                                                'label5',
                                                'gold_label'])
        elif args.data_sign == 'QNLI':
            rows = read_rows(input_file, names=['index', 'question', 'sentence', 'label'])
        pars = fp1.readlines()
        sents = []
        modi_sents = []
        if args.data_sign == 'SST-2':
            targets = rows['sentence']
        elif args.data_sign == 'MRPC':
            targets = rows["#1 String"]
        elif args.data_sign == 'MNLI':
            targets = rows["sentence2"]
        elif args.data_sign == 'QNLI':
            targets = rows['sentence']
        for idx, sent in targets.items():
            sents.append(sent)
        cnt = 0
        rate_sum = 0
        with open('./{}_text.txt'.format(args.data_sign), 'w', encoding='utf-8') as ftt:
            for sent, par in zip(sents, pars):
                ftt.write(sent.strip() + '\n')
                cnt += 1
                sent_len = len(sent.strip().split(' '))
                error_num = math.ceil(sent_len * float(rate))
                idx_lists, error_type_lst = find_pos(error_matrix, sent, par, error_num)

                # Set p if some positions should be excluded, for example entities in NER task
                p = list()
                # Variable modi can be used to trace the error positions.
                sent, modi = change_sent(sent, idx_lists, error_type_lst, error_matrix, preps, dets, trans, p)
                modi_sents.append(' '.join(sent))
                cnt_m = 0
                for i in modi:
                    if not i == '0':
                        cnt_m += 1
                rate_sum += cnt_m / len(sent)
                while '' in sent:
                    sent.remove('')
                ftt.write(' '.join(sent) + '\n')
                ftt.write('\n')

        print('rate', rate_sum / cnt)


        if args.data_sign == 'MRPC':
            modi_rows = {"Quality": [], "#1 ID": [], "#2 ID": [], "#1 String": [], "#2 String": []}
            for idx, row in rows.iterrows():
                modi_rows["Quality"].append(row["Quality"])
                modi_rows["#1 ID"].append(row["#1 ID"])
                modi_rows["#2 ID"].append(row["#2 ID"])
                modi_rows["#1 String"].append(modi_sents[idx])
                modi_rows["#2 String"].append(row["#2 String"])
            modi_rows = pd.DataFrame(modi_rows, columns=["Quality", "#1 ID", "#2 ID", "#1 String", "#2 String"])
            modi_rows.to_csv(output_tsv, sep='\t', index=False, quoting=3)
        elif args.data_sign == 'MNLI':
            modi_rows = {'index': [], 'captionID': [], 'pairID': [], 'genre': [], 'sentence1_binary_parse': [],
                         'sentence2_binary_parse': [],
                         'sentence1_parse': [], 'sentence2_parse': [], 'sentence1': [], 'sentence2': [], 'label1': [],
                         'label2': [], 'label3': [],
                         'label4': [], 'label5': [], 'gold_label': []}
            for idx, row in rows.iterrows():
                modi_rows["index"].append(row["index"])
                modi_rows["promptID"].append(row["promptID"])
                modi_rows["pairID"].append(row["pairID"])
                modi_rows["genre"].append(row["genre"])
                modi_rows["sentence1_binary_parse"].append(row["sentence1_binary_parse"])
                modi_rows["sentence2_binary_parse"].append(row["sentence2_binary_parse"])
                modi_rows["sentence1_parse"].append(row["sentence1_parse"])
                modi_rows["sentence2_parse"].append(row["sentence2_parse"])
                modi_rows["sentence1"].append(row["sentence1"])
                modi_rows["sentence2"].append(modi_sents[idx])
                modi_rows["label1"].append(row["label1"])
                modi_rows["label2"].append(row["label2"])
                modi_rows["label3"].append(row["label3"])
                modi_rows["label4"].append(row["label4"])
                modi_rows["label5"].append(row["label5"])
                modi_rows["gold_label"].append(row["gold_label"])
            modi_rows = pd.DataFrame(modi_rows, columns=['index', 'captionID', 'pairID', 'genre', 'sentence1_binary_parse',
                                                         'sentence2_binary_parse',
                                                         'sentence1_parse', 'sentence2_parse', 'sentence1', 'sentence2',
                                                         'label1', 'label2', 'label3',
                                                         'label4', 'label5', 'gold_label'])
            modi_rows.to_csv(output_tsv, sep='\t', index=False, quoting=3)
        elif args.data_sign == 'SST-2':
            modi_rows = {'sentence': [], 'label': []}
            for idx, row in rows.iterrows():
                modi_rows["sentence"].append(modi_sents[idx])
                modi_rows["label"].append(row["label"])
            modi_rows = pd.DataFrame(modi_rows,
                                     columns=['sentence', 'label'])
            modi_rows.to_csv(output_tsv, sep='\t', index=False, quoting=3)

        elif args.data_sign == 'QNLI':
            modi_rows = {'index': [], 'question': [], 'sentence': [], 'label': []}
            for idx, row in rows.iterrows():
                modi_rows["index"].append(row["index"])
                modi_rows["question"].append(row["question"])
                modi_rows["sentence"].append(modi_sents[idx])
                modi_rows["label"].append(row["label"])
            modi_rows = pd.DataFrame(modi_rows,
                                     columns=['index', 'question', 'sentence', 'label'])
            modi_rows.to_csv(output_tsv, sep='\t', index=False, quoting=3)




if __name__ == '__main__':
    if sys.argv[1] == 'csv':
        run(sys.argv[2:])
