# coding=gb2312
from graphs_extractor import preprocessing
from slither import Slither
from slither.printers.functions import cfg
import logging
from slither import Slither
# from graphviz import Source
import os
import re


# get for looping termination condition
def get_for_termination_condition(sol_path, func_name):
    pre_contract = preprocessing.get_contract(sol_path)
    flag = -1
    for i, line in enumerate(pre_contract.split('\n')):
        if func_name in line and 'function' in line:
            flag = i
    code = None
    for i in range(flag + 1, len(pre_contract.split('\n'))):
        if 'for' in pre_contract.split('\n')[i]:
            code = pre_contract.split('\n')[i]
            break
    if code is not None:
        match = re.search(r'\((.*?)\)', code)
        if match:
            code = (match.group(1)).split('; ')
            # if code[-1].startswith(' '):
            #     code[-1].lstrip()
            return code[-1]


def get_cfg_by_func(sol_path, buggy_function_name):
    node_list = []
    start_list = []
    target_list = []
    edge_labels = []
    graphs = {}
    slither = Slither(sol_path)
    for contract in slither.contracts:
        function_indices = [i for i, x in enumerate(contract.functions) if str(x) in buggy_function_name]
        for x in contract.functions:
            if str(x) in buggy_function_name:
                buggy_function_name.remove(str(x))
        for index in function_indices:
            function = contract.functions[index]
            for node in function.nodes:
                if [node.type, node.expression] not in node_list:
                    if str(node.type) == 'ENTRY_POINT':
                        # node_list.append([node.type, function])
                        function_state = str(function.visibility) + ' ' + str(function.payable)
                        node_list.append([function, function_state])
                    else:
                        node_list.append([node.type, node.expression])
            if_node = None
            if_loop_node = None
            for_if_loop_node = None
            for_if_loop_flag = 0
            for i in range(1, len(function.nodes)):
                j = i - 1
                front_node = function.nodes[j]
                back_node = function.nodes[i]
                if if_node is None and if_loop_node is None and for_if_loop_node is None:
                    if str(front_node.type) == 'ENTRY_POINT':
                        function_state = str(function.visibility) + ' ' + str(function.payable)
                        start_list.append(node_list.index([function, function_state]))
                    else:
                        start_list.append(node_list.index([front_node.type, front_node.expression]))
                    target_list.append(node_list.index([back_node.type, back_node.expression]))
                    edge_labels.append([None, None])
                if if_node is not None:
                    if front_node == if_node:
                        start_list.append(node_list.index([front_node.type, front_node.expression]))
                        target_list.append(node_list.index([back_node.type, back_node.expression]))
                        edge_labels.append(['IF', True])
                    elif str(back_node.type) == 'IF':
                        start_list.append(node_list.index([if_node.type, if_node.expression]))
                        target_list.append(node_list.index([back_node.type, back_node.expression]))
                        edge_labels.append(['IF', False])
                        if_node = back_node
                    elif str(front_node.type) == 'RETURN':
                        start_list.append(node_list.index([if_node.type, if_node.expression]))
                        target_list.append(node_list.index([back_node.type, back_node.expression]))
                        edge_labels.append(['IF', False])
                        if_node = None
                    elif str(back_node.type) == 'END_IF':
                        start_list.append(node_list.index([front_node.type, front_node.expression]))
                        target_list.append(node_list.index([back_node.type, back_node.expression]))
                        edge_labels.append([None, None])
                    elif str(front_node.type) == 'END_IF':
                        start_list.append(node_list.index([front_node.type, front_node.expression]))
                        target_list.append(node_list.index([back_node.type, back_node.expression]))
                        edge_labels.append([None, None])
                        start_list.append(node_list.index([if_node.type, if_node.expression]))
                        target_list.append(node_list.index([back_node.type, back_node.expression]))
                        edge_labels.append(['IF', False])
                        if_node = None
                    else:
                        start_list.append(node_list.index([front_node.type, front_node.expression]))
                        target_list.append(node_list.index([back_node.type, back_node.expression]))
                        edge_labels.append([None, None])
                if if_loop_node is not None:
                    if front_node == if_loop_node:
                        start_list.append(node_list.index([front_node.type, front_node.expression]))
                        target_list.append(node_list.index([back_node.type, back_node.expression]))
                        edge_labels.append(['IF_LOOP', True])
                    elif str(front_node.type) == 'END_LOOP':
                        start_list.append(node_list.index([if_loop_node.type, if_loop_node.expression]))
                        target_list.append(node_list.index([back_node.type, back_node.expression]))
                        edge_labels.append(['IF_LOOP', False])
                        if_loop_node = None
                    elif str(back_node.type) == 'END_LOOP':
                        start_list.append(node_list.index([front_node.type, front_node.expression]))
                        target_list.append(node_list.index([if_loop_node.type, if_loop_node.expression]))
                        edge_labels.append(['WHILE', None])
                    else:
                        start_list.append(node_list.index([front_node.type, front_node.expression]))
                        target_list.append(node_list.index([back_node.type, back_node.expression]))
                        edge_labels.append([None, None])
                if for_if_loop_node is not None:
                    # get for looping termination condition
                    condition = get_for_termination_condition(sol_path, str(function))
                    if front_node == for_if_loop_node:
                        start_list.append(node_list.index([front_node.type, front_node.expression]))
                        target_list.append(node_list.index([back_node.type, back_node.expression]))
                        edge_labels.append(['FOR_IF_LOOP', True])
                    elif str(front_node.expression) == condition:  # get for loop condition
                        start_list.append(node_list.index([front_node.type, front_node.expression]))
                        target_list.append(node_list.index([for_if_loop_node.type, for_if_loop_node.expression]))
                        edge_labels.append(['FOR', None])
                        start_list.append(node_list.index([for_if_loop_node.type, for_if_loop_node.expression]))
                        target_list.append(node_list.index([back_node.type, back_node.expression]))
                        edge_labels.append(['FOR_IF_LOOP', False])
                        for_if_loop_node = None
                        for_if_loop_flag = 0
                    else:
                        start_list.append(node_list.index([front_node.type, front_node.expression]))
                        target_list.append(node_list.index([back_node.type, back_node.expression]))
                        edge_labels.append([None, None])

                if str(back_node.type) == 'IF':
                    if_node = back_node
                if str(front_node.type) == 'BEGIN_LOOP' and str(back_node.type) == 'IF_LOOP':
                    if_loop_node = back_node
                if str(front_node.type) == 'BEGIN_LOOP' and str(back_node.type) == 'END_LOOP':
                    for_if_loop_flag = 1
                if for_if_loop_flag == 1:
                    if str(back_node.type) == 'IF_LOOP':
                        for_if_loop_node = back_node
            if node_list:
                graphs[str(function)] = {'node_list': node_list, 'start_list': start_list,
                                         'target_list': target_list, 'edge_labels': edge_labels}
            node_list = []
            start_list = []
            target_list = []
            edge_labels = []

    return graphs


def get_buggy_function_name(sol_path, log_path):
    buggy_function_name = []
    buggy_function = []
    bug_log = []
    with open(sol_path, 'r', encoding='utf-8') as sol_file:
        sol = sol_file.readlines()
    with open(log_path, 'r', encoding='utf-8') as log_file:
        log = log_file.readlines()
    for line in log[1:]:
        bug_log.append(line.split(',')[0: 2])
    bug_log = sorted(bug_log, key=lambda x: int(x[0]))
    for line in bug_log:
        buggy_function.append(''.join(sol[int(line[0]) - 1: int(line[0]) - 1 + int(line[1])]))
    for function in buggy_function:
        pattern = r'function[^\n](\w+)\s*\(.*'
        matches = re.findall(pattern, function)
        buggy_function_name.extend(match for match in matches)
    unique_list = []
    for item in buggy_function_name:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list


# get all bug function in a sol file
def get_buggy_func(sol_path, log_path):
    buggy_function_name = get_buggy_function_name(sol_path, log_path)
    return get_cfg_by_func(sol_path, buggy_function_name)


