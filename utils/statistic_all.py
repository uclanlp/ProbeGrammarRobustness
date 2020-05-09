# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from pattern3.en import tenses, conjugate
import inflection
from collections import defaultdict


def dis():
    dir_path = '/home/yinfan/conll14st-test-data/noalt/'
    file_path = os.path.join(dir_path, 'official-2014.combined.m2')
    avg_errors_per_sent = list()
    errors_sum_dict = dict()
    cnt_sent = 0
    cnt_error = 0
    fine_grained_dict = {"false_plural": 0,
                         "false_singular": 0,
                         "false_3sg": 0,
                         "false_n3sg": 0,
                         "false_vt": 0,
                         "false_wform": 0,
                         "false_tran": 0,
                         "false_woinc": 0,
                         "false_woadv": 0,
                         "false_prep": 0,
                         "false_art": 0,
                         }
    preps = defaultdict(lambda: 0.0)
    dets = defaultdict(lambda: 0.0)
    trans = defaultdict(lambda: 0.0)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if line and line[0] == 'S':
                cnt_sent += 1
                avg_errors_per_sent.append(cnt_error)
                cnt_error = 0
                last_sent = line[1:]
            elif line and line[0] == 'A':
                start = int(line[1])
                line2 = line[2].split('|||')
                end = int(line2[0])
                errortype = line2[1]
                subset = line2[2].lower() if line2[2] else ''
                if errortype in errors_sum_dict:
                    errors_sum_dict[errortype] += 1
                else:
                    errors_sum_dict[errortype] = 1
                if errortype == 'Trans':
                    fine_grained_dict['false_tran'] += 1
                    if end == start:
                        inw = ''
                    elif end == start + 1:
                        inw = last_sent[start].lower()
                    else:
                        inw = (' '.join(last_sent[start:end])).lower()
                    trans[subset, inw] += 1
                if errortype == "ArtOrDet":
                    fine_grained_dict['false_art'] += 1
                    if end == start:
                        inw = ''
                    elif end == start + 1:
                        inw = last_sent[start].lower()
                    else:
                        inw = (' '.join(last_sent[start:end])).lower()
                    dets[subset, inw] += 1
                if errortype == "Prep":
                    fine_grained_dict['false_prep'] += 1
                    if end == start:
                        inw = ''
                    elif end == start + 1:
                        inw = last_sent[start].lower()
                    else:
                        inw = (' '.join(last_sent[start:end])).lower()
                    preps[subset, inw] += 1
                if errortype == "Vt" or errortype == "Vform":
                    fine_grained_dict['false_vt'] += 1
                if errortype == "Wci":
                    fine_grained_dict['false_wform'] += 1
                if errortype == "WOinc":
                    fine_grained_dict['false_woinc'] += 1
                if errortype == "WOadv":
                    fine_grained_dict['false_woadv'] += 1
                if errortype == "Nn":
                    if end == start + 1:
                        inw = last_sent[start].lower()
                    else:
                        continue
                    if inflection.pluralize(inw) == inw:
                        fine_grained_dict['false_plural'] += 1
                    else:
                        fine_grained_dict['false_singular'] += 1
                if errortype == "SVA":
                    if end == start + 1:
                        inw = last_sent[start].lower()
                    else:
                        continue
                    if '3sg' in tenses(subset) or '3sgp' in tenses(subset):
                        fine_grained_dict['false_n3sg'] += 1
                    elif('lsg' in tenses(subset) or '2sg' in tenses(subset)):
                        fine_grained_dict['false_3sg'] += 1
                    elif('lsgp' in tenses(subset) or '2sgp' in tenses(subset)):
                        fine_grained_dict['false_3sg'] += 1
                cnt_error += 1

    # Statistics of coarse error type count
    sorted_errors_sum_dict = sorted(errors_sum_dict.items(), key=lambda x: x[1], reverse=True)
    print('sorted_error_dict', sorted_errors_sum_dict)
    print(len(sorted_errors_sum_dict))

    # 0 for lex, 1 for sent level
    false_levels = [0 for i in range(2)]

    # 0 for lex, 1 for sent set
    false_sets = [[] for i in range(2)]

    false_sets[0] = ['false_vt', 'false_wform', 'false_art', 'false_plural', 'false_singular', 'false_prep']
    false_sets[1] = ['false_3sg', 'false_n3sg', 'false_tran', 'false_woadv', 'false_woinc']

    for i in range(len(false_levels)):
        for type in false_sets[i]:
            false_levels[i] += fine_grained_dict[type]

    total = sum(false_levels)
    print(false_levels)
    coarse_dis = [fal / total for fal in false_levels]
    fine_dis = [[0 for i in range(len(false_sets[j]))] for j in range(len(false_levels))]
    for i in range(len(false_levels)):
        for j in range(len(false_sets[i])):
            fine_dis[i][j] = fine_grained_dict[false_sets[i][j]] / false_levels[i]
    print(coarse_dis)
    print(fine_dis)

    return coarse_dis, fine_dis, false_sets

def stat():
    dir_path = '/home/yinfan/conll14st-test-data/noalt/'
    file_path = os.path.join(dir_path, 'official-2014.combined.m2')

    avg_errors_per_sent = list()
    errors_sum_dict = dict()
    cnt_sent = 0
    cnt_error = 0
    preps = defaultdict(lambda: 0.0)
    dets = defaultdict(lambda: 0.0)
    trans = defaultdict(lambda: 0.0)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if line and line[0] == 'S':
                cnt_sent += 1
                avg_errors_per_sent.append(cnt_error)
                cnt_error = 0
                last_sent = line[1:]
            elif line and line[0] == 'A':
                start = int(line[1])
                line2 = line[2].split('|||')
                end = int(line2[0])
                errortype = line2[1]
                subset = line2[2].lower() if line2[2] else ''
                if errortype in errors_sum_dict:
                    errors_sum_dict[errortype] += 1
                else:
                    errors_sum_dict[errortype] = 1
                if errortype == 'Trans':
                    if end == start:
                        inw = ''
                    elif end == start + 1:
                        inw = last_sent[start].lower()
                    else:
                        inw = (' '.join(last_sent[start:end])).lower()
                    trans[subset, inw] += 1
                if errortype == "ArtOrDet":
                    if end == start:
                        inw = ''
                    elif end == start + 1:
                        inw = last_sent[start].lower()
                    else:
                        inw = (' '.join(last_sent[start:end])).lower()
                    dets[subset, inw] += 1
                if errortype == "Prep":
                    if end == start:
                        inw = ''
                    elif end == start + 1:
                        inw = last_sent[start].lower()
                    else:
                        inw = (' '.join(last_sent[start:end])).lower()
                    preps[subset, inw] += 1

    return preps, dets, trans

if __name__ == '__main__':
    main()

