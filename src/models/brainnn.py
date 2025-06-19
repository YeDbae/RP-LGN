import torch
from collections import defaultdict
import numpy as np 
from itertools import permutations
from torch_geometric.utils import to_dense_adj
from torch.nn import functional as F


class BrainNN(torch.nn.Module):
    """def __init__(self, args, gnn, discriminator=lambda x, y: x @ y.t()):
        super(BrainNN, self).__init__()
        self.gnn = gnn
        self.pooling = args.pooling
        self.discriminator = discriminator""" # 原来的init

    """def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        g, s1 = self.gnn(x, edge_index, edge_attr, batch)
        log_logits = F.log_softmax(g, dim=-1)

        return log_logits, s1"""  # 原来的forward 以上就是全部，要是裂开了请把下面的全部删掉

    def __init__(self, args, gnn1, discriminator=lambda x, y: x @ y.t()):
        super(BrainNN, self).__init__()
        self.gnn1 = gnn1
        # self.pooling = args.pooling
        self.discriminator = discriminator

    def forward(self, data):
        x, edge_index, edge_attr,  batch, y = data.x, data.edge_index, data.edge_attr,  data.batch,data.y
        out, loss_set, att_tsne= self.gnn1(x, edge_index, edge_attr, batch)
        #print(out)
        # log_logits = F.log_softmax(g_gat, dim=-1)
        out = F.log_softmax(out, dim=-1)
        #print(out)
        
        return out, loss_set,att_tsne
