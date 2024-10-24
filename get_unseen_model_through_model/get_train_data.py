# coding=gb2312
# import pandas as pd
import torch
# from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, DistilBertTokenizer, DistilBertModel
# import transformers
import os
# from get_unseen_model_through_model import model
from GNN_model import net
import pickle


def get_buggy_function(sol_path, csv_path):
    buggy_function = []
    with open(sol_path, encoding='utf8') as sol_file:
        sol = sol_file.readlines()
    for i, line in enumerate(sol):
        sol[i] = line.strip()
    with open(csv_path, encoding='utf8') as csv_file:
        csv = csv_file.readlines()
    for i in range(1, len(csv)):
        start = int(csv[i].split(',')[0]) - 1
        target = int(csv[i].split(',')[0]) + int(csv[i].split(',')[1])
        buggy_function.append(''.join(sol[start: target]))
    return buggy_function


def get_all_bug_code(code_dir, bug):
    bugs = []
    code_dir = os.path.join(code_dir, bug)
    for i in range(1, 51):
        sol_path = os.path.join(code_dir, f'buggy_{i}.sol')
        csv_path = os.path.join(code_dir, f'BugLog_{i}.csv')
        bugs += get_buggy_function(sol_path, csv_path)
    return bugs


def load_data(description_path, model_dir_path, code_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = DistilBertTokenizer.from_pretrained('./distilbert-base-uncased')
    with open(description_path, 'r') as description_txt:
        lines = description_txt.readlines()
    # data = []
    buggy_types = []
    key_codes = []
    reasons = []
    results = []
    attack_methods = []
    solutions = []

    for line in lines:
        if 'buggy_type' in line:
            buggy_types.append(line.strip())
        if 'key_code' in line:
            key_codes.append(line.strip())
        if 'reason' in line:
            reasons.append(line.strip())
        if 'result' in line:
            results.append(line.strip())
        if 'attack_method' in line:
            attack_methods.append(line.strip())
        if 'solution' in line:
            solutions.append(line.strip())

    parameters = {}
    pth = torch.load('../trained_model/Reentrancy.pth')
    for name in pth:
        parameters[name] = []

    buggy_types_data = []
    key_codes_data = []
    reasons_data = []
    results_data = []
    attack_methods_data = []
    solutions_data = []
    code_data = []

    for (buggy_type, key_code, reason, result, attack_method, solution) in zip(buggy_types, key_codes, reasons, results,
                                                                               attack_methods, solutions):
        bug = buggy_type.split(': ')[1]
        pth_path = os.path.join(model_dir_path, f'{bug}.pth')
        trained_model = torch.load(pth_path)
        for code in get_all_bug_code(code_dir, bug):
            buggy_types_data.append(buggy_type)
            key_codes_data.append(key_code)
            reasons_data.append(reason)
            results_data.append(result)
            attack_methods_data.append(attack_method)
            solutions_data.append(solution)
            code_data.append(code)

            for name in trained_model:
                parameters[name].append(trained_model[name])

    buggy_types_tokenized = tokenizer(buggy_types_data, padding=True, truncation=True, return_tensors='pt')
    key_codes_tokenized = tokenizer(key_codes_data, padding=True, truncation=True, return_tensors='pt')
    reasons_tokenized = tokenizer(reasons_data, padding=True, truncation=True, return_tensors='pt')
    results_tokenized = tokenizer(results_data, padding=True, truncation=True, return_tensors='pt')
    attack_methods_tokenized = tokenizer(attack_methods_data, padding=True, truncation=True, return_tensors='pt')
    solutions_tokenized = tokenizer(solutions_data, padding=True, truncation=True, return_tensors='pt')
    code_tokenized = tokenizer(code_data, padding=True, truncation=True, return_tensors='pt')

    # get actual length
    buggy_types_valid_lens = []
    key_codes_valid_lens = []
    reasons_valid_lens = []
    results_valid_lens = []
    attack_methods_valid_lens = []
    solutions_valid_lens = []
    code_valid_lens = []
    for buggy_types_mask, key_codes_mask, reasons_mask, results_mask, attack_methods_mask, solutions_mask, code_mask \
            in zip(buggy_types_tokenized['attention_mask'].tolist(),
                   key_codes_tokenized['attention_mask'].tolist(),
                   reasons_tokenized['attention_mask'].tolist(),
                   results_tokenized['attention_mask'].tolist(),
                   attack_methods_tokenized['attention_mask'].tolist(),
                   solutions_tokenized['attention_mask'].tolist(),
                   code_tokenized['attention_mask'].tolist()):

        buggy_types_valid_lens.append(buggy_types_mask.count(1))
        key_codes_valid_lens.append(key_codes_mask.count(1))
        reasons_valid_lens.append(reasons_mask.count(1))
        results_valid_lens.append(results_mask.count(1))
        attack_methods_valid_lens.append(attack_methods_mask.count(1))
        solutions_valid_lens.append(solutions_mask.count(1))
        code_valid_lens.append(code_mask.count(1))

    buggy_types_valid_lens = torch.tensor(buggy_types_valid_lens)
    key_codes_valid_lens = torch.tensor(key_codes_valid_lens)
    reasons_valid_lens = torch.tensor(reasons_valid_lens)
    results_valid_lens = torch.tensor(results_valid_lens)
    attack_methods_valid_lens = torch.tensor(attack_methods_valid_lens)
    solutions_valid_lens = torch.tensor(solutions_valid_lens)
    code_valid_lens = torch.tensor(code_valid_lens)

    for name in parameters:
        parameters[name] = torch.stack(parameters[name])
    # parameters = torch.tensor(parameters)

    data_array = ()
    data_array += (buggy_types_tokenized['input_ids'], buggy_types_tokenized['attention_mask'], buggy_types_valid_lens)
    data_array += (key_codes_tokenized['input_ids'], key_codes_tokenized['attention_mask'], key_codes_valid_lens)
    data_array += (reasons_tokenized['input_ids'], reasons_tokenized['attention_mask'], reasons_valid_lens)
    data_array += (results_tokenized['input_ids'], results_tokenized['attention_mask'], results_valid_lens)
    data_array += (attack_methods_tokenized['input_ids'], attack_methods_tokenized['attention_mask'],
                   attack_methods_valid_lens)
    data_array += (solutions_tokenized['input_ids'], solutions_tokenized['attention_mask'], solutions_valid_lens)
    data_array += (code_tokenized['input_ids'], code_tokenized['attention_mask'], code_valid_lens)
    data_array += tuple(parameters.values())

    # save data_array
    torch.save({'data_array': data_array}, './data_array/data_array.pth')
    print('save successfully!!!')


if __name__ == '__main__':
    description_path = '../semantic_spaces/seen_buggy_description.txt'
    model_dir_path = '../trained_model'
    code_dir = '../data/buggy_contracts'
    load_data(description_path, model_dir_path, code_dir)

