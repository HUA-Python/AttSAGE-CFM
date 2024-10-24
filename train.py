# coding=gb2312
import os
import random
import sys
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.loader import DataLoader
from GNN_model import net
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec
import time
from torch.utils.tensorboard import SummaryWriter
from data_set.dataset_extractor import GetCFGDataset
import random


def train(model, crit, optim, dataset, loader, epoch, device, train_type):
    model.train()
    loss_all = 0
    with tqdm(loader, file=sys.stdout) as bar:
        for data in bar:
            data.to(device)
            optim.zero_grad()
            out = model(data)
            y = data.y
            loss = crit(out, y)
            loss_all += data.num_graphs * loss.item()
            if train_type == 'train':
                loss.backward()
                optim.step()
                bar.set_description('epoch:{}'.format(epoch))
                time.sleep(0.1)
    return loss_all / len(dataset)


def evalute(loader, model, device):
    model.eval()
    prediction = []
    labels = []
    with torch.no_grad():
        with tqdm(loader, file=sys.stdout) as testbar:
            for data in testbar:
                data.to(device)
                pred = model(data)
                label = data.y
                prediction.append(pred)
                labels.append(label)
                testbar.set_description('test')
    prediction = np.hstack([pred.cpu().numpy() for pred in prediction])
    labels = np.hstack([label.cpu().numpy() for label in labels])
    return roc_auc_score(labels, prediction)


def train_model_by_func(data_dir, buggy_name, root):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = []
    test_dataset = []
    bug_type_list = os.listdir(data_dir)

    all_dataset = {}
    dataset_len_sum = 0
    bug_flag = bug_type_list.index(buggy_name)
    for buggy in bug_type_list:
        y = bug_type_list.index(buggy)
        one_buggy = GetCFGDataset(data_dir, buggy, y, root=root)
        if buggy != buggy_name:
            dataset_len_sum += len(one_buggy)
        all_dataset[buggy] = one_buggy
    for buggy_type in all_dataset:
        buggy_dataset = all_dataset[buggy_type]
        # get index for training
        if buggy_type != buggy_name:
            data_num = int(len(buggy_dataset) / dataset_len_sum * len(all_dataset[buggy_name]))
            data_index = random.sample(range(len(buggy_dataset)), data_num)
        else:
            data_index = range(len(buggy_dataset))

        train_index = data_index[0: int(len(data_index) * 0.8)]
        test_index = data_index[int(len(data_index) * 0.8):]
        for i in train_index:
            train_data = buggy_dataset[i]
            if train_data.y.item() == bug_flag:
                train_data.y = torch.FloatTensor([1])
            else:
                train_data.y = torch.FloatTensor([0])
            train_dataset.append(train_data)
        for i in test_index:
            test_data = buggy_dataset[i]
            if test_data.y.item() == bug_flag:
                test_data.y = torch.FloatTensor([1])
            else:
                test_data.y = torch.FloatTensor([0])
            test_dataset.append(test_data)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    model = net.Model()
    # model = net.GlobalPoolModel()
    model.to(device)
    crit = torch.nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    # draw picture
    writer = SummaryWriter(f'./train_logs/{buggy_name}_logs')
    # write result in file
    res_file = open('result.txt', 'a')

    best_train_loss = 1
    # best_train_auc = 0
    best_test_loss = 1
    best_test_auc = 0
    best_model = None

    num_epochs = 50

    for epoch in range(num_epochs):
        print('---train---')
        train_loss = train(model, crit, optim, train_dataset, train_loader, epoch, device, 'train')
        if train_loss < best_train_loss:
            best_train_loss = train_loss

        # train_auc = evalute(train_loader, model, device)
        # if train_auc > best_train_auc:
        #     best_train_auc = train_auc
        print('---test---')
        test_loss = train(model, crit, optim, test_dataset, test_loader, epoch, device, 'test')
        if test_loss < best_test_loss:
            best_test_loss = test_loss

        test_auc = evalute(test_loader, model, device)
        if test_auc > best_test_auc:
            best_test_auc = test_auc

        if test_auc >= best_test_auc and test_loss <= best_test_loss:
            best_model = model

        # draw fig
        writer.add_scalars("fig", {'train_loss': train_loss, 'test_loss': test_loss, 'test_auc': test_auc}, epoch + 1)

        print('train_loss: {}; test_loss: {}; test_auc: {}'.format(train_loss, test_loss, test_auc))

    writer.close()
    print('===================')
    print('best_train_loss: {}; best_test_loss: {}; best_test_auc: {}'.
          format(best_train_loss, best_test_loss, best_test_auc))
    # write result in file
    res_file.write(f'{buggy_name}:' + '\n' + f'best_train_loss: {best_train_loss};'
                                             f'best_test_loss: {best_test_loss}; best_test_auc: {best_test_auc}' + '\n')
    res_file.close()

    # save model
    torch.save(best_model.state_dict(), f'./trained_model/{buggy_name}.pth')
    # load model
    # model = net.Model()
    # model.load_state_dict(torch.load('./trained_model/best_RE_model.pth'))


if __name__ == '__main__':
    data_dir = './data/buggy_contracts'
    buggy_name = 'OverflowUnderflow'
    root = './data_set/dataset/'
    bug_list = os.listdir(data_dir)
    for buggy_name in bug_list:
        train_model_by_func(data_dir, buggy_name, root)
    # train_model_by_func(data_dir, buggy_name, root)

