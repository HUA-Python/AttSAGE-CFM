# coding=gb2312
from transformers import DistilBertTokenizer
import torch
from get_unseen_model_through_model import model
from GNN_model import net
import os


def get_model_parameter(unseen_bug, unseen_description_path, model_pth_path):
    tokenizer = DistilBertTokenizer.from_pretrained('./distilbert-base-uncased')
    with open(unseen_description_path, 'r') as description_txt:
        lines = description_txt.readlines()

    buggy_type = ''
    key_code = ''
    reason = ''
    result = ''
    attack_method = ''
    solution = ''
    code = ''

    for line in lines:
        if buggy_type is not '' and 'buggy_type' in line:
            break
        elif buggy_type is not '' and line.split(': ')[0] == 'key_code':
            key_code = line.strip()
        elif buggy_type is not '' and line.split(': ')[0] == 'reason':
            reason = line.strip()
        elif buggy_type is not '' and line.split(': ')[0] == 'result':
            result = line.strip()
        elif buggy_type is not '' and line.split(': ')[0] == 'attack_method':
            attack_method = line.strip()
        elif buggy_type is not '' and line.split(': ')[0] == 'solution':
            solution = line.strip()
        elif buggy_type is not '' and line.split(': ')[0] == 'code':
            code = line.strip()
        elif 'buggy_type' in line and unseen_bug in line and buggy_type is '':
            buggy_type = line.strip()

    buggy_type_tokenized = tokenizer(buggy_type, padding=True, truncation=True, return_tensors='pt')
    key_code_tokenized = tokenizer(key_code, padding=True, truncation=True, return_tensors='pt')
    reason_tokenized = tokenizer(reason, padding=True, truncation=True, return_tensors='pt')
    result_tokenized = tokenizer(result, padding=True, truncation=True, return_tensors='pt')
    attack_method_tokenized = tokenizer(attack_method, padding=True, truncation=True, return_tensors='pt')
    solution_tokenized = tokenizer(solution, padding=True, truncation=True, return_tensors='pt')
    code_tokenized = tokenizer(code, padding=True, truncation=True, return_tensors='pt')

    Xs = [buggy_type_tokenized['input_ids'], key_code_tokenized['input_ids'], reason_tokenized['input_ids'],
          result_tokenized['input_ids'], attack_method_tokenized['input_ids'], solution_tokenized['input_ids'],
          code_tokenized['input_ids']]
    attention_masks = [buggy_type_tokenized['attention_mask'], key_code_tokenized['attention_mask'],
                       reason_tokenized['attention_mask'], result_tokenized['attention_mask'],
                       attack_method_tokenized['attention_mask'], solution_tokenized['attention_mask'],
                       code_tokenized['attention_mask']]

    valid_lens = []
    for attention_mask in attention_masks:
        valid_len = []
        valid_len.append(torch.squeeze(attention_mask, dim=0).tolist().count(1))
        valid_lens.append(torch.tensor(valid_len))

    # load model
    num_layers, dropout = 4, 0.2

    enc_num_hiddens = 768
    num_hiddens = 128
    enc_ffn_num_input, enc_ffn_num_hiddens, enc_num_heads = 768, 1536, 4
    enc_key_size, enc_query_size, enc_value_size = 768, 768, 768
    enc_norm_shape = [768]
    output_shapes = [(128, 128), (128,), (128, 128), (1, 128), (128, 128), (128,), (128, 128), (1, 128), (1,), (1, 128),
                     (1, 4, 32), (1, 4, 32), (1, 4, 32), (128,), (128, 128), (128, 128), (128, 7), (1, 128), (1,),
                     (1, 128), (128, 128), (128,), (64, 128), (64,), (1, 64), (1,)]

    encoder = model.Encoder()
    transformer_encoder = model.TransformerEncoder(enc_key_size, enc_query_size, enc_value_size, enc_num_hiddens,
                                                   enc_norm_shape, enc_ffn_num_input, enc_ffn_num_hiddens,
                                                   enc_num_heads, num_layers, dropout)
    decoder = model.Decoder(transformer_encoder)
    net = model.EncoderDecoder(encoder, decoder, num_hiddens, output_shapes)
    net.load_state_dict(torch.load(model_pth_path))

    Y_hat = net(Xs, attention_masks, valid_lens)
    return Y_hat


def get_unseen_model(unseen_bug, unseen_description_path, model_pth_path):
    model = net.Model()
    # print(pth_list)
    params = get_model_parameter(unseen_bug, unseen_description_path, model_pth_path)
    pth = torch.load('../trained_model/Reentrancy.pth')
    parameters = {}
    for parameter_name, parameter_value in zip(pth.keys(), params):
        parameters[parameter_name] = torch.squeeze(parameter_value, dim=0)
    for name, param in model.named_parameters():
        param.data.copy_(parameters[name])
    torch.save(model.state_dict(), f'../unseen_model/{unseen_bug}.pth')


if __name__ == '__main__':
    unseen_bug = 'UncheckedSend'
    unseen_description_path = '../semantic_spaces/unseen_buggy_description.txt'
    model_pth_path = './transformer_model/transformer.pth'
    get_unseen_model(unseen_bug, unseen_description_path, model_pth_path)

