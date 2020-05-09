# !/usr/bin/env python
# -*- coding: utf-8 -*-

class Node(object):  # Node for n-ary tree

    def __init__(self, value, children):
        self.value = value
        self.children = children

class Tree(object):

    def __init__(self, root):
        self.root = root

    def find_det_ins(self, root, idx_list):

        if root.value == "NP":
            valid = True
            for child in root.children:
                if child.value == "DT" or child.value == "NP" or child.value == "PRP$" or child.value == "PRP":
                    valid = False
            if valid:
                idx_list.append(self.get_idx(root))
        for child in root.children:
            self.find_det_ins(child, idx_list)


    def find_sng_nouns(self, root, idx_list):
        if root.value == "NN":
            idx_list.append(self.get_idx(root))
        for child in root.children:
            self.find_sng_nouns(child, idx_list)

    def find_pl_noun(self, root, idx_list):
        if root.value == "NNS":
            idx_list.append(self.get_idx(root))
        for child in root.children:
            self.find_pl_noun(child, idx_list)

    def find_prep_ins(self, root, idx_list):
        if root.value == "VP":
            for child in root.children:
                if child.value == "NP":
                    idx_list.append(self.get_idx(child))
        for child in root.children:
            self.find_prep_ins(child, idx_list)

    def find_worder(self, root, idx_list):
        if root.value == "VP":
            if (root.children[0].value == "MD" or "VB" in root.children[0].value) and (root.children[1].value == "ADVP" or root.children[1].value == "ADVJ"):
                idx_list.append(self.get_idx(root.children[0]))
        for child in root.children:
            self.find_worder(child, idx_list)


    def find_advp(self, root, idx_list):
        if root.value == "ADVP":
            if len(root.children) == 1 and root.children[0].value == "RB" and not self.get_idx(root) == 0:
                idx_list.append(self.get_idx(root.children[0]))
            if len(root.children) == 2 and root.children[0].value in ["RBR", "RBS"]:
                idx_list.append(self.get_idx(root.children[0]))
        if root.value == "ADJP":
            if len(root.children) == 2 and root.children[0].value in ["RBR", "RBS"]:
                idx_list.append(self.get_idx(root.children[0]))
            if root.children[1] == "JJ" and root.children[0] == "RB":
                idx_list.append(self.get_idx(root.children[0]))
            if root.children[0] == "JJ" and root.children[1] == "RB":
                idx_list.append(self.get_idx(root.children[0]))
            if root.children[0] == "RB"  and "VB" in root.children[1]:
                idx_list.append(self.get_idx(root.children[0]))
        for child in root.children:
            self.find_advp(child, idx_list)


    def find_verb(self, root, idx_list):
        if root.value == "VB" or root.value == "VBZ" or root.value == "VBP" or root.value == "VBD" or root.value == "VBN" or root.value == "VBG":
            idx_list.append(self.get_idx(root))
        for child in root.children:
            self.find_verb(child, idx_list)

    def find_wform(self, root, idx_list):
        if root.value == "JJ" or root.value == "VB" or root.value == "VBZ" or root.value == "VBP" or root.value == "VBD" or root.value == "VBN" or root.value == "VBG":
            idx_list.append(self.get_idx(root))
        for child in root.children:
            self.find_wform(child, idx_list)

    def find_3sg(self, root, idx_list):
        if root.value == "VBZ":
            idx_list.append(self.get_idx(root))
        for child in root.children:
            self.find_3sg(child, idx_list)


    def find_n3sg(self, root, idx_list):
        if root.value == "VBP" or root.value == "VB":
            idx_list.append(self.get_idx(root))
        for child in root.children:
            self.find_n3sg(child, idx_list)

    def get_idx(self, root):
        if len(root.children):
            return self.get_idx(root.children[0])
        else:
            return root.value[1]