import torch.nn
from torch_geometric.nn import TopKPooling, SAGEConv, GATConv, SAGPooling, GCNConv
from torch_geometric.nn import global_mean_pool as gap
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(128, 128)
        self.pooling1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pooling2 = SAGPooling(128)

        self.conv3 = GATConv(128, 32, 4, dropout=0.5, edge_dim=7)
        self.pooling3 = SAGPooling(128)

        self.linear1 = torch.nn.Linear(128, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pooling1(x, edge_index, edge_attr, batch)
        # 全局特征向量
        x1 = gap(x, batch)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pooling2(x, edge_index, edge_attr, batch)
        # 全局特征向量
        x2 = gap(x, batch)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pooling3(x, edge_index, edge_attr, batch)
        # 全局特征向量
        x3 = gap(x, batch)

        x = x1+x2+x3

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x.squeeze(1)

