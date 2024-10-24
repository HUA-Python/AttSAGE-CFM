import os
import csv
import re
from random import random
import chardet
from collections import Counter
import solcx
from slither import Slither
from tqdm import tqdm


def get_encoding(file):
    # get byte data
    with open(file, 'rb') as f:
        return chardet.detect(f.read())['encoding']


# delete the comment in contract
def remove_comments(input_code):
    # delete one-line comment
    code_without_single_line_comments = re.sub(r'\/\/[^\n]*', '', input_code)
    # delete multi-line comments
    code_without_comments = re.sub(r'\/\*[\s\S]*?\*\/', '', code_without_single_line_comments)
    return code_without_comments


def remove_whitespace(input_string):
    # remove space
    lines = [line.strip() for line in input_string.splitlines() if line.strip()]
    result = "\n".join(lines)
    return result


def get_contract(sol_path):
    with open(sol_path, 'r', encoding=get_encoding(sol_path)) as f:
        contract = f.read()
        contract = remove_whitespace(remove_comments(contract))
        return contract


# get all function and parameter name
def get_parameters_functions(children, parameters, functions):
    parameter_error = ['balance', 'value', 'length', 'sender', 'timestamp', 'data', '_', 'sig', 'gasprice']
    function_error = ['transfer', 'add', 'send', 'sub']
    for child in children:
        if child['name'] == 'VariableDeclaration':
            if child['attributes']['name'] is not None and child['attributes']['name'] != '' \
                    and child['attributes']['name'] not in parameter_error:
                if 'mapping' in child['attributes']['type']:
                    parameters[child['attributes']['name']] = 'mapping'
                else:
                    parameters[child['attributes']['name']] = child['attributes']['type']
        if child['name'] == 'FunctionDefinition':
            if child['attributes']['name'] is not None and child['attributes']['name'] != '' \
                    and child['attributes']['name'] not in function_error:
                functions.append(child['attributes']['name'])
        if 'children' in child:
            parameters, functions = get_parameters_functions(child['children'], parameters, functions)
    return parameters, functions


# normalize sol file
def normalize_sol(sol_path, copy_path, solc_version='0.5.11'):
    encoding = get_encoding(sol_path)
    sol_file = open(sol_path, 'r+', encoding='utf-8').read()
    solcx.install_solc(solc_version)
    compiled_sol = solcx.compile_files([sol_path], output_values=["ast"], solc_version=solc_version)
    parameters = {}
    functions = []
    # get all function and parameter name in sol
    for contract in compiled_sol.values():
        get_parameters_functions(contract['ast']['children'], parameters, functions)
    # set normalization name for all function and parameter name
    data_type_num = {'uint': 0, 'bytes': 0, 'bool': 0, 'address': 0, 'string': 0, 'mapping': 0}
    function_flag = 0
    function_list = {}
    parameter_list = {}
    functions = list(dict.fromkeys(functions))
    for parameter in parameters:
        for data_type in data_type_num:
            if data_type in parameters[parameter]:
                data_type_num[data_type] += 1
                parameter_list[parameter] = f'{data_type}_{data_type_num[data_type]}'

    for function in functions:
        function_flag += 1
        function_list[function] = f'function_{function_flag}'
    # replace function and parameter name
    function_list.update(parameter_list)
    for function in function_list:
        sol_file = re.sub(r'(?<![a-zA-Z0-9_])' + re.escape(function) + r'(?![a-zA-Z0-9_])', function_list[function],
                          sol_file)
    with open(copy_path, 'w', encoding='utf-8') as file:
        file.write(sol_file)

    # return the maps of actual function names and the names after changing
    function_list = {value: key for key, value in function_list.items()}
    return function_list


data_dir = '../data/buggy_contracts'


def normalize_all_sol(data_dir):
    type_list = os.listdir(data_dir)
    for buggy_type in type_list:
        type_path = os.path.join(data_dir, buggy_type)
        with tqdm(range(1, 51)) as bar:
            for i in bar:
                sol_path = os.path.join(type_path, f'buggy_{i}.sol')
                normalize_sol(sol_path, sol_path)

