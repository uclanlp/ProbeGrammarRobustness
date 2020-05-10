import numpy as np
from collections import defaultdict
import codecs
import copy
import subprocess
import sys
import argparse
# Vt           Verb tense
# Vform        Verb form
# SVA          Subject-verb-agreement
# ArtOrDet     Article or Determiner
# Nn           Noun number
# Prep         Preposition
# WOinc        Incorrect sentence form
# WOadv        Adverb/adjective position
# Trans        Link word/phrases

def main(args):
    # Read the file
    with codecs.open("/home/yinfan/conll14st-test-data/noalt/official-2014.combined.m2", 'r', 'utf-8') as inp:
        lines = inp.readlines()
    positions = []
    insert_dict = []
    flag = 1
    cnt = 0
    # Parse file and collect counts
    with codecs.open("./{}_cor.txt".format(args.type), 'w', 'utf-8') as fc, codecs.open("./{}_err.txt".format(args.type), 'w', 'utf-8') as fe:
        for line in lines:
            if line and line[0] == 'S':
                if cnt > 0 and flag == 1:
                    for idx, bias in enumerate(insert_dict[:-1]):
                        for i, poss in enumerate(positions_cor[idx + 1:]):
                            for pos in range(len(poss)):
                                positions_cor[idx + i + 1][pos] = str(int(positions_cor[idx + i + 1][pos]) + bias)
                    positions_cor_bk = []
                    for i in range(len(positions_cor)):
                        for j in range(len(positions_cor[i])):
                            positions_cor_bk.append(positions_cor[i][j])
                    error_sent = ' '.join(last_sent)
                    correct_sent = ' '.join(correct_last_sent)
                    fe.write(error_sent + '\t' + ' '.join(positions))
                    fe.write('\n')
                    fc.write(correct_sent + '\t' + ' '.join(positions_cor_bk))
                    fc.write('\n')
                line = line.strip().split()
                last_sent = copy.deepcopy(line[1:])
                correct_last_sent = copy.deepcopy(line[1:])
                positions = []
                positions_cor = []
                insert_dict = []
                cnt += 1
                flag = 0
            elif line and line[0] == 'A':
                line = line.strip().split('|||')
                start = int(line[0].split()[1])
                end = int(line[0].split()[2])
                errortype = line[1].strip()
                if not errortype == args.type:
                    continue
                subst = line[2].lower() if line[2] else ''
                if not (end == start + 1):
                    continue
                if not (len(subst.strip().split()) == 1):
                    continue
                flag = 1
                if not subst == '':
                    correct_last_sent = correct_last_sent[:start] + [subst] + correct_last_sent[end:]
                else:
                    print(correct_last_sent)
                    correct_last_sent = correct_last_sent[:start] + correct_last_sent[end:]
                    print(correct_last_sent)
                positions += [str(i) for i in range(int(start), int(end))]
                subst_lst = subst.strip().split(' ')
                err_len = end - start
                cor_len = len(subst_lst)
                if subst == '':
                    cor_len = 0
                positions_cor.append([str(i) for i in range(int(start), int(start) + len(subst_lst))])
                insert_dict.append(cor_len - err_len)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='Prep', type=str)
    args = parser.parse_args()
    main(args)
