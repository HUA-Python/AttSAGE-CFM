# coding=gb2312
import math
# from d2l import torch as d2l
from transformers import BertModel, DistilBertModel
import torch
import torch.nn as nn
from functools import reduce
import operator
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

    def forward(self, X, attention_mask):
        outputs = self.bert(input_ids=X, attention_mask=attention_mask)
        return outputs[0]


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # The masked elements on the last axis are replaced with a very large negative value
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weight = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weight), values)


class PositionalEncoding(nn.Module):
    """location enciding"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))

        position = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        div_term = torch.exp(
            torch.arange(0, num_hiddens, 2).float() * -(torch.log(torch.tensor(10000.0)) / num_hiddens))

        self.P[:, :, 0::2] = torch.sin(position * div_term)

        if num_hiddens % 2 == 1:
            self.P[:, :, 1::2] = torch.cos(position * div_term[:-1])
        else:
            self.P[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


def transpose_qkv(X, num_heads):
    """transfer the shape"""
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """reverse the transpose_qkv"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.w_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.w_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.w_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.w_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.w_q(queries)
        queries = transpose_qkv(queries, self.num_heads)
        keys = transpose_qkv(self.w_k(keys), self.num_heads)
        values = transpose_qkv(self.w_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.w_o(output_concat)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_output, **kwargs):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_output)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout,
                                            bias=use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module('block' + str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                                              ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        X = self.pos_encoding(X * math.sqrt(self.num_hiddens))
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)

        return X


class Decoder(nn.Module):
    def __init__(self, transformer_encoder, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.transformer_encoder = transformer_encoder

    def forward(self, enc_X, *args):
        enc_outputs = self.transformer_encoder(enc_X, *args)
        return enc_outputs


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, num_hiddens, output_shapes, **kwargs):
        super(EncoderDecoder, self).__init__()
        self.buggy_type_encoder = encoder
        self.buggy_type_decoder = decoder

        self.key_codes_encoder = encoder
        self.key_codes_decoder = decoder

        self.reasons_encoder = encoder
        self.reasons_decoder = decoder

        self.results_encoder = encoder
        self.results_decoder = decoder

        self.attack_methods_encoder = encoder
        self.attack_methods_decoder = decoder

        self.solutions_encoder = encoder
        self.solutions_decoder = decoder

        self.code_encoder = encoder
        self.code_decoder = decoder

        self.linear = nn.Linear(768, 768)
        self.relu = torch.nn.ReLU()
        self.linear_1 = nn.Linear(768, 512)
        self.linear_2 = nn.Linear(512, 128)

        self.linear_layers = nn.ModuleList([nn.Linear(num_hiddens, int(torch.prod(torch.tensor(shape))))
                                            for shape in output_shapes])
        self.output_shapes = output_shapes

    def forward(self, Xs, attention_masks, valid_lens):
        results = []

        for i, (X, attention_mask, valid_len) in enumerate(zip(Xs, attention_masks, valid_lens)):
            if i == 0:
                output = self.buggy_type_encoder(X, attention_mask)
                result = self.buggy_type_decoder(output, valid_len)
            elif i == 1:
                output = self.key_codes_encoder(X, attention_mask)
                result = self.key_codes_decoder(output, valid_len)
            elif i == 2:
                output = self.reasons_encoder(X, attention_mask)
                result = self.reasons_decoder(output, valid_len)
            elif i == 3:
                output = self.results_encoder(X, attention_mask)
                result = self.results_decoder(output, valid_len)
            elif i == 4:
                output = self.attack_methods_encoder(X, attention_mask)
                result = self.attack_methods_decoder(output, valid_len)
            elif i == 5:
                output = self.solutions_encoder(X, attention_mask)
                result = self.solutions_decoder(output, valid_len)
            elif i == 6:
                output = self.code_encoder(X, attention_mask)
                result = self.code_decoder(output, valid_len)
            results.append(result)

        linear_result = [self.relu(self.linear(tensor)) for tensor in results]
        X = torch.cat(linear_result, dim=1)

        X = self.relu(self.linear_1(X))
        X = self.relu(self.linear_2(X))

        outputs = []
        for i, linear_layer in enumerate(self.linear_layers):
            output = linear_layer(X[:, 0, :])  # Use the representation of the first token (class token)
            output = output.view(-1, *self.output_shapes[i])
            outputs.append(output)
        return outputs

