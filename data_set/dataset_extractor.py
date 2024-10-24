# coding=gb2312
import math
import os
import random
import re
import sys
import numpy as np
from gensim.models import Word2Vec
from sklearn import manifold
from slither import Slither
from tqdm import tqdm
from torch_geometric.data import Data
from graphs_extractor import get_cfg
from graphs_extractor import preprocessing
from torch_geometric.nn import Node2Vec
import torch
from torch_geometric.data import InMemoryDataset, Dataset
import shutil
from graphs_extractor.get_cfg import get_buggy_func


# get tokens
def tokenize(text):
    special_chars = "(){},[];."
    for char in special_chars:
        text = text.replace(char, f" {char} ")
    return text.split(' ')


# get word2vec model
def train_word2vec_model(data_dir):
    train_data = []
    type_list = os.listdir(data_dir)
    for buggy_type in type_list:
        type_path = os.path.join(data_dir, buggy_type)

        with tqdm(range(1, 51)) as bar:
            for i in bar:
                sol_path = os.path.join(type_path, f'buggy_{i}.sol')
                slither = Slither(sol_path)
                for contract in slither.contracts:
                    for function in contract.functions:
                        function_tokens = []
                        for node in function.nodes:
                            text = str(node)
                            if str(node.type) == 'ENTRY_POINT':
                                text = str(function) + ' ' + str(function.visibility) + ' ' + str(function.payable)
                                # text = str(node) + ' ' + str(function)
                            function_tokens.extend(tokenize(text))
                        if function_tokens is not None:
                            train_data.append(function_tokens)
                bar.set_description(buggy_type)


    word2vec_model = Word2Vec(train_data, vector_size=128, window=2, min_count=1, workers=4, epochs=800)
    word2vec_model.wv.save_word2vec_format('./word2vec_model/data.vector', binary=False)
    word2vec_model.save('./word2vec_model/word2vec.model')
    # return train_data


# transfer edge_label to vector
def get_edge_label_vec(edge_labels):
    label_mapping = {False: 0, True: 1, 'FOR': 2, 'FOR_IF_LOOP': 3, 'WHILE': 4, 'IF': 5, 'IF_LOOP': 6}
    vectors = []
    for label_pair in edge_labels:
        vector = np.zeros(len(label_mapping))
        for label in label_pair:
            if label is not None:
                vector[label_mapping[label]] = 1
        vectors.append(vector.tolist())
    return torch.tensor(vectors, dtype=torch.float)


# transfer code to vector through word2vec
def average_vec(tokens, word2vec_model):
    vec = torch.zeros(word2vec_model.vector_size)
    tokens_num = 0
    for token in tokens:
        # if token not in word2vec_model.wv:
        #     print(token, '---', tokens)
        if token in word2vec_model.wv:
            vec += word2vec_model.wv[token]
            tokens_num += 1
    if tokens_num > 0:
        return vec / tokens_num
    else:
        return None


# transfer the one-line code into tensor
def get_node_vec(node_list, word2vec_model):
    for i, node in enumerate(node_list):
        if str(node[1]) == 'None':
            tokens = tokenize(str(node[0]))
        else:
            tokens = tokenize(str(node[0])) + tokenize(str(node[1]))
        vec = average_vec(tokens, word2vec_model)
        if vec is not None:
            node_list[i] = vec.tolist()
    return torch.FloatTensor(node_list)


class GetCFGDataset(InMemoryDataset):
    def __init__(self, data_dir, buggy_name, y, root, transform=None, pre_transform=None):
        self.data_dir = data_dir
        self.buggy_name = buggy_name
        self.y = y
        super(GetCFGDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'cfg_{self.buggy_name}.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []
        sol_dir_path = os.path.join(self.data_dir, self.buggy_name)
        file_list = os.listdir(sol_dir_path)
        sol_list = [file for file in file_list if file.endswith('.sol')]
        # log_list = [file for file in file_list if file.endswith('.csv')]
        word2vec_model = Word2Vec.load('./word2vec_model/word2vec.model')
        with tqdm(range(1, len(sol_list) + 1)) as bar:
            for i in bar:
                sol_path = os.path.join(sol_dir_path, f'buggy_{i}.sol')
                log_path = os.path.join(sol_dir_path, f'BugLog_{i}.csv')
                function_graphs = get_buggy_func(sol_path, log_path)
                for function_graph in function_graphs:
                    node_list = function_graphs[function_graph]['node_list']
                    start_list = function_graphs[function_graph]['start_list']
                    target_list = function_graphs[function_graph]['target_list']
                    edge_labels = function_graphs[function_graph]['edge_labels']
                    x = get_node_vec(node_list, word2vec_model)
                    edge_index = torch.tensor([start_list, target_list], dtype=torch.long)
                    edge_attr = get_edge_label_vec(edge_labels)
                    # pre_data_list.append({'x': x, 'edge_index': edge_index, 'dege_attr': edge_attr})
                    data = Data(x=x, y=torch.FloatTensor([self.y]), edge_index=edge_index, edge_attr=edge_attr)
                    data_list.append(data)
                bar.set_description(f'{self.buggy_name}')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def get_all_datasets(data_dir, root):
    bug_type_list = os.listdir(data_dir)
    for buggy_name in bug_type_list:

        y = bug_type_list.index(buggy_name)
        GetCFGDataset(data_dir, buggy_name, y, root=root)

