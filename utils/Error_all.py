# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import inflection
from collections import defaultdict
from pattern3.en import tenses, conjugate, lexeme
from nltk.corpus import wordnet
from statistic_all import stat, dis

class Errors(object):
    def __init__(self, preps, dets, trans):
        self.confusion_matrix = {}
        self.idx2error = {'0': "MEC", "1": "ART", "2": "FAL_S", "3": "FAL_P", "4": "FAL_3SG", "5": "FRAG", "6": "PREP",
                     "7": "FAL_N3SG", "8": "TRAN", "9": "VT", "10": "VFORM", "11": "WORDER", "12": "WFORM", "13": "WOADV"}
        self.error2idx = {value: key for key, value in self.idx2error.items()}
        self.punctuation = [".", ",", ":", "?", "!", "\"", ";", "\'"]
        self.init_confusion_matrix(self.idx2error, self.confusion_matrix)
        self.init_statistics(preps, dets, trans)
        self.init_distribution()

    def init_distribution(self):
        self.coarse_grained_dis, self.fine_grained_dis, self.false_sets = dis()

    def init_statistics(self, preps, dets, trans):
        self.prepsum, self.prepvals, self.prepps, self.preppps = self.prep_statistics(preps)
        self.detsum, self.detvals, self.detps, self.detpps = self.det_statistics(dets)
        self.transsum, self.transvals, self.transps, self.transpps = self.trans_statistics(trans)

    def init_confusion_matrix(self, idx2error, confusion_matrix):
        for key, value in idx2error.items():
            if value == 'PREP':
                confusion_matrix[value] = ['on', 'in', 'at', 'from', 'for', 'under', 'over', 'with', 'into', 'during',
                                           'until', 'against', 'among', 'throughout','of', 'to', 'by', 'about', 'like',
                                           'before', 'after', 'since', 'across', 'behind', 'but', 'out', 'up', 'down', 'off', '']
            elif value == 'ART':
                confusion_matrix[value] = ['a', 'an', 'the', '']
            elif value == 'TRAN':
                confusion_matrix[value] = ['and', 'but', 'so', 'however', 'as', 'that', 'thus', 'also', 'because',
                                           'therefore', 'if', 'although', 'which', 'where', 'moreover', 'besides',
                                           'although', 'of', '']
        self.confusion_matrix['WH'] = ['what', 'which', 'that']
        self.confusion_matrix['WH$'] = ['who', 'that']

    def capitalize(self, s):
        if s:
            s = s[0].upper() + s[1:]
        return s

    def decapitalize(self, s):
        if s:
            s = s[0].lower() + s[1:]
        return s

    def det_statistics(self, dets):
        detlist = self.confusion_matrix['ART']
        detsum = defaultdict(lambda: 0.0)
        for d in detlist:
            for e in detlist:
                if e != d:
                    detsum[d] += dets[d, e]

        detvals = defaultdict(lambda: [])
        detps = defaultdict(lambda: [])
        for d in detlist:
            for e in detlist:
                if dets[d, e] and d != e:
                    detvals[d].append(e)
                    detps[d].append(dets[d, e] / detsum[d])

        detpps = {}
        alldets = np.sum([detsum[d] for d in detsum])
        for d in detps:
            detpps[d] = detsum[d] / alldets

        return detsum, detvals, detps, detpps

    def trans_statistics(self, trans):
        translist = self.confusion_matrix['TRAN']
        transsum = defaultdict(lambda: 0.0)
        for a in translist:
            for b in translist:
                if trans[a, b] and a != b:
                    transsum[a] += trans[a, b]

        transvals = defaultdict(lambda: [])
        transps = defaultdict(lambda: [])
        for d in translist:
            for e in translist:
                if trans[d, e] and d != e:
                    transvals[d].append(e)
                    transps[d].append(trans[d, e] / transsum[d])

        transpps = defaultdict(lambda: 0.0)
        alltrans = np.sum([transsum[d] for d in transsum])
        for d in transps:
            transpps[d] = transsum[d] / alltrans
        return transsum, transvals, transps, transpps


    def prep_statistics(self, preps):
        preplist = self.confusion_matrix['PREP']
        prepsum = defaultdict(lambda: 0.0)
        for a in preplist:
            for b in preplist:
                if preps[a, b] and a != b:
                    prepsum[a] += preps[a, b]

        prepvals = defaultdict(lambda: [])
        prepps = defaultdict(lambda: [])
        for d in preplist:
            for e in preplist:
                if preps[d, e] and d != e:
                    prepvals[d].append(e)
                    prepps[d].append(preps[d, e] / prepsum[d])

        preppps = defaultdict(lambda: 0.0)
        allpreps = np.sum([prepsum[d] for d in prepsum])
        for d in prepps:
            preppps[d] = prepsum[d] / allpreps
        return prepsum, prepvals, prepps, preppps

    def intro_art_error(self, line, pos_lst, dets, p):
        modi = (0, 0)
        if pos_lst or not pos_lst:
            ays = line.count('a') + line.count('A')
            ans = line.count('an') + line.count('An')
            thes = line.count('the') + line.count('The')
            # Count possible insertions
            pos_ins_count = len(pos_lst)
            pos = list()
            # If there are some possible article errors
            if ays + ans + thes + pos_ins_count:
                # Convert the counts to probabilities and
                tempsum = float(
                    ays * self.detpps['a'] + ans * self.detpps['an'] + thes * self.detpps['the'] + pos_ins_count * self.detpps[''])
                vals = ['a', 'an', 'the', '']
                # ... make a probability distribution
                ps = [ays * self.detpps['a'] / tempsum, ans * self.detpps['an'] / tempsum, thes * self.detpps['the'] / tempsum,
                      pos_ins_count * self.detpps[''] / tempsum]

                # ... and sample from it
                ch = np.random.choice(vals, 1, p=ps)[0]
                # Sample an error
                subst = np.random.choice(self.detvals[ch], 1, p=self.detps[ch])[0]
                # Sample a position, in case the same article exists in more than one positions
                if ch == '':
                    pos = pos_lst
                else:
                    pos = [index for index, word in enumerate(line) if (word == ch or word == self.capitalize(ch)) and (index not in p)]
            if not pos:
                return modi
            ind = np.random.choice(pos, 1)[0]
            while ind in pos_lst:
                pos_lst.remove(ind)

            # If we are substituting
            if subst and ch != '':
                if line[ind][0] == line[ind][0].upper():
                    subst = self.capitalize(subst)
                line[ind] = subst
                modi = (ind, 1)
            # If we are inserting
            elif subst and ch == '':
                if subst in line[ind]:
                    return modi
                if ind == 0:
                    subst = self.capitalize(subst)
                line[ind] = subst + ' ' + self.decapitalize(line[ind])

                modi = (ind, 2)
            # Else if we are deleting an article
            else:
                line[ind] = ''
                if ind == 0:
                    line[ind + 1] = self.capitalize(line[ind + 1])
                modi = (ind, 3)
            p.append(ind)
        return modi

    def intro_prep_error(self, line, pos_lst, preps, p):
        modi = (0, 0)
        if pos_lst or not pos_lst:
            count_occs = []
            pos = list()
            # Count how many times each preposition exists in the line
            for d in self.confusion_matrix["PREP"][:-1]:
                count_occs.append(line.count(d) + line.count(self.capitalize(d)))
            count_occs.append(len(pos_lst))
            # Multiply those counts with the probabilities if these prepositions (from the corpus)
            tempsum = np.sum([count_occs[i] * self.preppps[d] for i, d in enumerate(self.confusion_matrix["PREP"])])
            # Now, if there exist some prepositions in the line
            if tempsum > 0:
                vals = self.confusion_matrix["PREP"]
                # Create probability distribution over the prepositions of the sentence
                ps = [count_occs[i] * self.preppps[d] / tempsum for i, d in enumerate(self.confusion_matrix["PREP"])]
                # Randomly sample from said distribution
                ch = np.random.choice(vals, 1, p=ps)[0]

                # Now randomly sample the error, according to which preposition we chose to substitute
                subst = np.random.choice(self.prepvals[ch], 1, p=self.prepps[ch])[0]
                # Find the list of the positions where the preposition exists in the sentence (could be more than one!)
                if ch == '':
                    pos = pos_lst
                else:
                    pos = [index for index, word in enumerate(line) if (word == ch or word == self.capitalize(ch)) and (index not in p)]
            if not pos:
                return modi
            # Randomly choose one of these positions (we only want to subtsitute one of them)
            ind = np.random.choice(pos, 1)[0]
            while ind in pos_lst:
                pos_lst.remove(ind)
            # If we are substituting (not deleting)
            if subst and ch != '':
                if line[ind][0] == line[ind][0].upper():
                    subst = self.capitalize(subst)
                line[ind] = subst
                modi = (ind, 1)
            # If we are inserting
            elif subst and ch == '':
                if subst in line[ind]:
                    return modi
                if ind == 0:
                    subst = self.capitalize(subst)
                line[ind] = subst + ' ' + self.decapitalize(line[ind])
                modi = (ind, 2)
            # Else if we are deleting
            else:
                line[ind] = ''
                if ind == 0:
                    line[ind + 1] = self.capitalize(line[ind + 1])
                modi = (ind, 3)
            p.append(ind)
        return modi

    def all_punct(self, a):
        for c in a:
            if c not in self.punctuation:
                return True
        return False

    def intro_wform_error(self, line, pos_lst):
        modi = (0, 0)
        if pos_lst:
            ind = np.random.choice(pos_lst, 1)[0]
            while ind in pos_lst:
                pos_lst.remove(ind)
            if len(line[ind]) > 1:
                add = ""
                backup = line[ind]
                while line[ind] and line[ind][-1] in self.punctuation:
                    add = line[ind][-1] + add
                    line[ind] = line[ind][:-1]
                synonyms = []
                for syn in wordnet.synsets(line[ind]):
                    for lm in syn.lemmas():
                        synonyms.append(lm.name())
                synonyms = list(set(synonyms))
                for syn in synonyms:
                    if len(syn.split(' ')) > 1:
                        synonyms.remove(syn)
                if len(synonyms) == 0:
                    line[ind] = backup
                    return modi
                line[ind] = np.random.choice(synonyms, 1)[0]
                if line[ind]:
                    line[ind] = line[ind] + add
                    modi = (ind, 1)
                else:
                    line[ind] = backup
        return modi

    def intro_vt_error(self, line, pos_lst):
        modi = (0, 0)
        if pos_lst:
            ind = np.random.choice(pos_lst, 1)[0]
            while ind in pos_lst:
                pos_lst.remove(ind)
            if len(line[ind]) > 1:
                add = ""
                backup = line[ind]
                while line[ind] and line[ind][-1] in self.punctuation:
                    add = line[ind][-1] + add
                    line[ind] = line[ind][:-1]
                replace_lst = lexeme(line[ind])
                while line[ind] in replace_lst:
                    replace_lst.remove(line[ind])
                for rep in replace_lst:
                    if len(rep.split(' ')) > 1 or "n't" in rep:
                        replace_lst.remove(rep)
                if len(replace_lst) == 0:
                    line[ind] = backup
                    return modi
                line[ind] = np.random.choice(replace_lst, 1)[0]
                if line[ind]:
                    line[ind] = line[ind] + add
                    modi = (ind, 1)
                else:
                    line[ind] = backup
        return modi

    def intro_frag_error(self, line, pos_lst):
        modi = (0, 0)
        if pos_lst:
            ind = pos_lst[0]
            subst = np.random.choice(self.confusion_matrix['TRAN'][:-1], 1)[0]
            if subst in line[ind]:
                return modi
            if ind == 0:
                subst = self.capitalize(subst)
            line[ind] = subst + ' ' + self.decapitalize(line[ind])
            modi = (ind, 2)
        return modi

    def intro_worder_error(self, line, pos_lst):
        modi = (0, 0)
        if pos_lst:
            ind = np.random.choice(pos_lst, 1)[0]
            while ind in pos_lst:
                pos_lst.remove(ind)
            if ind == len(line) - 1:
                return modi
            else:
                line[ind], line[ind + 1] = line[ind + 1], line[ind]
                modi = [(ind, 1), (ind + 1, 1)]
        return modi

    def intro_trans_error(self, line, pos_lst, trans, p):
        modi = (0, 0)
        if pos_lst:
            count_occs = []

            for d in self.confusion_matrix["TRAN"][:-1]:
                count_occs.append(line.count(d) + line.count(self.capitalize(d)))
            count_occs.append(len(pos_lst))

            tempsum = np.sum([count_occs[i] * self.transpps[d] for i, d in enumerate(self.confusion_matrix["TRAN"])])

            if tempsum > 0:
                vals = self.confusion_matrix["TRAN"]

                ps = [count_occs[i] * self.transpps[d] / tempsum for i, d in enumerate(self.confusion_matrix["TRAN"])]

                ch = np.random.choice(vals, 1, p=ps)[0]

                subst = np.random.choice(self.transvals[ch], 1, p=self.transps[ch])[0]

                pos = [index for index, word in enumerate(line) if (word == ch or word == self.capitalize(ch)) and (index not in p)]
            if not pos:
                return modi

            ind = np.random.choice(pos, 1)[0]
            while ind in pos_lst:
                pos_lst.remove(ind)

            if subst and ch != '':
                if line[ind][0] == line[ind][0].upper():
                    subst = self.capitalize(subst)
                line[ind] = subst
                modi = (ind, 1)

            else:
                if len(line) == 1:
                    return modi
                line[ind] = ''

                if ind == 0:
                    line[ind + 1] = self.capitalize(line[ind + 1])
                modi = (ind, 3)
            p.append(ind)
        return modi

    def intro_nn_error(self, sent, pos_lst, type='p'):
        modi = (0, 0)
        if pos_lst:
            ind = np.random.choice(pos_lst, 1)[0]
            while ind in pos_lst:
                pos_lst.remove(ind)
            try:
                not_all_punct = self.all_punct(sent[ind])
            except:
                ind -= 1
                not_all_punct = self.all_punct(sent[ind])
            if sent[ind] and not_all_punct:
                add = ""
                while sent[ind][-1] in self.punctuation:
                    add = sent[ind][-1] + add
                    sent[ind] = sent[ind][:-1]
                if type == 'p':
                    sent[ind] = inflection.singularize(sent[ind])
                else:
                    sent[ind] = inflection.pluralize(sent[ind])
                sent[ind] = sent[ind] + add
                modi = (ind, 1)
        return modi

    def intro_mec_error(self, sent, pos_lst):
        modi = (0, 0)
        if not pos_lst:
            return modi
        pos = pos_lst[0]
        try:
            N = len(sent[pos])
        except:
            pos_lst[0] -= 1
            pos = pos_lst[0]
            N = len(sent[pos])

        if N <= 1:
            sent[pos] = sent[pos] + self.confusion_matrix['MEC'][np.random.randint(1, 26)]
        else:
            ppos = np.random.randint(N)
            p = np.random.uniform(0, 1)
            if p < 0.25:
                sent[pos] = sent[pos][:ppos] + self.confusion_matrix['MEC'][0] + sent[pos][ppos + 1:]
            elif p > 0.25 and p < 0.5:
                sent[pos] = sent[pos][:ppos + 1] + self.confusion_matrix['MEC'][np.random.randint(1, 26)] + sent[pos][ppos + 1:]
            elif p >= 0.5 and p < 0.75:
                repos = np.random.randint(N - 1)
                repl = sent[pos][repos + 1] + sent[pos][repos]
                sent[pos] = sent[pos][:repos] + repl + sent[pos][repos + 2:]
            else:
                sent[pos] = sent[pos][:ppos] + self.confusion_matrix['MEC'][np.random.randint(1, 26)] + sent[pos][ppos + 1:]
        modi = (pos, 1)
        return modi

    def intro_sva_error(self, line, pos_lst, type='past'):
        modi = (0, 0)
        if pos_lst:
            ind = np.random.choice(pos_lst, 1)[0]
            while ind in pos_lst:
                pos_lst.remove(ind)
            if len(line[ind]) > 1:
                add = ""
                backup = line[ind]
                while line[ind] and line[ind][-1] in self.punctuation:
                    add = line[ind][-1] + add
                    line[ind] = line[ind][:-1]
                if type == '3sg':
                    line[ind] = conjugate(line[ind], '1sg', parse=True)
                elif type == 'n3sg':
                    line[ind] = conjugate(line[ind], '3sg', parse=True)
                elif type == 'past':
                    line[ind] = conjugate(line[ind], '1sgp', parse=True)
                if line[ind]:
                    line[ind] = line[ind] + add
                    modi = (ind, 1)
                else:
                    line[ind] = backup
        return modi

