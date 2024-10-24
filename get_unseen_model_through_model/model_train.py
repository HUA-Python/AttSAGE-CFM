# coding=gb2312
import os
import sys
import time
import model
import torch
import get_train_data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import utils
from tqdm import *
from GNN_model import net


def train_method(net, data_iter, lr, num_epochs, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    loss= nn.MSELoss()
    best_net = None
    min_loss = 1
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    writer = SummaryWriter("../train_logs/Transformer_logs")
    for epoch in range(num_epochs):
        with tqdm(data_iter, file=sys.stdout) as bar:
            for batch in data_iter:
                total_loss = torch.tensor(0.0, device=device)
                optimizer.zero_grad()

                buggy_type, buggy_type_mask, buggy_type_valid_len = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                key_codes, key_codes_mask, key_codes_valid_len = batch[3].to(device), batch[4].to(device), batch[5].to(device)
                reasons, reasons_mask, reasons_valid_len = batch[6].to(device), batch[7].to(device), batch[8].to(device)
                results, results_mask, results_valid_len = batch[9].to(device), batch[10].to(device), batch[11].to(device)
                attack_methods, attack_methods_mask, attack_methods_valid_len = batch[12].to(device), batch[13].to(device), batch[14].to(device)
                solutions, solutions_mask, solutions_valid_len = batch[15].to(device), batch[16].to(device), batch[17].to(device)
                code, code_mask, code_valid_len = batch[18].to(device), batch[19].to(device), batch[20].to(device)
                Xs = [buggy_type, key_codes, reasons, results, attack_methods, solutions, code]
                attention_masks = [buggy_type_mask, key_codes_mask, reasons_mask, results_mask, attack_methods_mask,
                                   solutions_mask, code_mask]
                valid_lens = [buggy_type_valid_len, key_codes_valid_len, reasons_valid_len, results_valid_len,
                              attack_methods_valid_len, solutions_valid_len, code_valid_len]

                Y_hat = net(Xs, attention_masks, valid_lens)
                Y = []
                for i in range(21, len(batch)):
                    Y.append(batch[i].to(device))
                for y_hat, y in zip(Y_hat, Y):
                    total_loss += loss(y_hat, y)
                total_loss.backward()
                optimizer.step()
                bar.set_description(f'train_Epoch [{epoch + 1}/{num_epochs}]')
                bar.set_postfix(train_loss=total_loss.item())
                time.sleep(0.0001)
                bar.update(1)
                if total_loss.item() < min_loss:
                    best_net = net
                    min_loss = total_loss.item()
        writer.add_scalars('transformer_train_fig', {'loss': total_loss.item()}, epoch + 1)
    return best_net, min_loss


def train():
    data_array = torch.load(f'./data_array/data_array.pth')['data_array']

    dataset = torch.utils.data.TensorDataset(*data_array)
    data_iter = torch.utils.data.DataLoader(dataset, 16, shuffle=True)

    lr, num_epochs = 0.00002, 15
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    # net.to(device)

    net, loss = train_method(net, data_iter, lr, num_epochs, device)
    torch.save(net.state_dict(), f'./transformer_model/transformer.pth')
    print(f'loss {loss:.5f}')


if __name__ == '__main__':
    train()
