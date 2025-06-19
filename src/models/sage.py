import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, MessagePassing
from torch.nn import Parameter
import numpy as np
from torch.nn import functional as F
from torch_geometric.nn.inits import glorot, zeros
from typing import Tuple
from torch import Tensor
from torch_geometric.nn import SAGEConv, knn
from torch import nn
import torch_geometric
import math
from torch_geometric.nn import TopKPooling
from .utils import create_dataset
from .vbll.layers.classification import DiscClassification
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph

EPS = 1e-10

class DiscVBLLMLP(nn.Module):
  def __init__(self, cfg):
    super(DiscVBLLMLP, self).__init__()

    self.params = nn.ModuleDict({
      'in_layer': nn.Linear(cfg.IN_FEATURES, cfg.HIDDEN_FEATURES),
      'core': nn.ModuleList([nn.Linear(cfg.HIDDEN_FEATURES, cfg.HIDDEN_FEATURES) for i in range(cfg.NUM_LAYERS)]),
      'out_layer': DiscClassification(cfg.HIDDEN_FEATURES, cfg.OUT_FEATURES, cfg.REG_WEIGHT, parameterization = cfg.PARAM, return_ood=cfg.RETURN_OOD, prior_scale=cfg.PRIOR_SCALE),
    })
    self.activations = nn.ModuleList([nn.ELU() for i in range(cfg.NUM_LAYERS)])
    self.cfg = cfg

  def forward(self, x):
    x = x.view(x.shape[0], -1)
    x = self.params['in_layer'](x)

    for layer, ac in zip(self.params['core'], self.activations):
      x = ac(layer(x))

    return self.params['out_layer'](x)

class train_cfg:
  NUM_EPOCHS = 1
  BATCH_SIZE = 16
  LR = 3e-3
  WD = 1e-4
  OPT = torch.optim.AdamW
  CLIP_VAL = 1
  VAL_FREQ = 1
  VBLL = True

class cfg:
    IN_FEATURES = 784
    HIDDEN_FEATURES = 128
    OUT_FEATURES = 2
    NUM_LAYERS = 2
    REG_WEIGHT = 1./1009
    PARAM = 'diagonal'
    RETURN_OOD = True
    PRIOR_SCALE = 1.

##############################################################################


def topk_loss(s, ratio):
    if ratio > 0.5:
        ratio = 1 - ratio
    s = s.sort(dim=1).values
    res = -torch.log(s[:, -int(s.size(1) * ratio):] + EPS).mean() - torch.log(
        1 - s[:, :int(s.size(1) * ratio)] + EPS).mean()
    return res


class MPGCNConv(SAGEConv):
    def __init__(self, in_channels, out_channels, edge_emb_dim: int, gcn_mp_type: str, bucket_sz: float,
                 normalize: bool = True, root_weight: bool = True, project: bool = False,bias: bool = True):
        super(MPGCNConv, self).__init__(in_channels=in_channels, out_channels=out_channels, aggr='add')

        self.edge_emb_dim = edge_emb_dim
        self.gcn_mp_type = gcn_mp_type
        self.bucket_sz = bucket_sz
        self.bucket_num = math.ceil(2.0 / self.bucket_sz)
        if gcn_mp_type == "bin_concate":
            self.edge2vec = nn.Embedding(self.bucket_num, edge_emb_dim)

        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj_t = None

        input_dim = out_channels
        self.edge_lin = torch.nn.Linear(input_dim, out_channels)

        self.reset_parameters()



class SAGE(torch.nn.Module):
    def __init__(self, input_dim, args, num_nodes, num_classes):
        super(SAGE, self).__init__()
        self.activation = torch.nn.ReLU()
        self.convs = torch.nn.ModuleList()
        self.pooling = args.pooling
        self.num_nodes = num_nodes
        # 改
        self.ratio = args.ratio

        hidden_dim = args.hidden_dim
        num_layers = args.n_GNN_layers
        edge_emb_dim = args.edge_emb_dim
        gcn_mp_type = args.gcn_mp_type
        bucket_sz = args.bucket_sz
        gcn_input_dim = input_dim

        for i in range(num_layers - 1):
            conv = torch_geometric.nn.Sequential('x, edge_index', [
                (MPGCNConv(gcn_input_dim, gcn_input_dim, edge_emb_dim, gcn_mp_type, bucket_sz, normalize=True, bias=True),
                 'x, edge_index -> x'),
                nn.Linear(gcn_input_dim, gcn_input_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm1d(gcn_input_dim)
            ])
            # gcn_input_dim = hidden_dim
            self.convs.append(conv)

        self.conv1 = torch_geometric.nn.Sequential('x, edge_index', [
                (MPGCNConv(gcn_input_dim, gcn_input_dim, edge_emb_dim, gcn_mp_type, bucket_sz, normalize=True, bias=True),
                 'x, edge_index -> x'),
                nn.Linear(gcn_input_dim, gcn_input_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm1d(gcn_input_dim)
            ])

        self.conv2 = torch_geometric.nn.Sequential('x, edge_index', [
                (MPGCNConv(gcn_input_dim, gcn_input_dim, edge_emb_dim, gcn_mp_type, bucket_sz, normalize=True, bias=True),
                 'x, edge_index -> x'),
                nn.Linear(gcn_input_dim, gcn_input_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm1d(gcn_input_dim)
            ])

        input_dim = 0


        self.convs.append(conv)

        self.fcn = nn.Sequential(
            nn.Linear(40000, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )

        self.vbmlp = nn.Sequential(
            nn.Linear(36100, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.vbllayer=DiscClassification(32, 2, cfg.REG_WEIGHT, parameterization = cfg.PARAM, return_ood=cfg.RETURN_OOD, prior_scale=cfg.PRIOR_SCALE)

        self.kk = nn.Sequential(nn.BatchNorm1d(3200))


    def forward(self, x, edge_index, edge_attr,batch):
        z = x

        for i, conv in enumerate(self.convs):
            print(i)
            # bz*nodes, hidden
            z = z+ conv(z, edge_index)


        edge_attr = torch.abs(edge_attr)

# Fully connected graph to one nearest neighbor (KNN, k=1) (except for themselves)
################################################################################################################################################

        knn_index = knn(z, z, k=2, batch_x=batch, batch_y=batch)

        neighbor_indices = knn_index[1, :]

        odd_indices_mask = torch.arange(neighbor_indices.shape[0]) % 2 == 1
        odd_neighbor_indices = neighbor_indices[odd_indices_mask]

        if z.device != odd_neighbor_indices.device:
            odd_neighbor_indices = odd_neighbor_indices.to(z.device)

        z_updated = z[odd_neighbor_indices]
        updated_edge_attr = edge_attr[odd_neighbor_indices]

        node_indices = torch.arange(0, len(batch), dtype=odd_neighbor_indices.dtype, device=odd_neighbor_indices.device)
        updated_indices_2d = torch.stack((node_indices.unsqueeze(0), odd_neighbor_indices.unsqueeze(0)), dim=0).squeeze(1)

        knn_data = Data(x=z_updated, edge_index=updated_indices_2d, edge_attr=updated_edge_attr)

        src = knn_data.edge_index[0]  # index of the source node
        tgt = knn_data.edge_index[1]  # index of the target node

        src_features = knn_data.x[src]  # feature of the source node [3200, 200]
        tgt_features = knn_data.x[tgt]  # feature of the target node [3200, 200]
        sum_node_features = src_features + tgt_features

        expanded_knn_edge_attr = knn_data.edge_attr.unsqueeze(1)

        # 200 node feature
        concatenated_knn_data = Data(x=z_updated, edge_index=updated_indices_2d, edge_attr=sum_node_features)


        #concatenated_edge_feature = torch.cat([expanded_knn_edge_attr, sum_node_features], dim=1)
        #concatenated_knn_data = Data(x=z_updated, edge_index=updated_indices_2d, edge_attr=concatenated_edge_feature)

# Constructing line graphs
################################################################################################################################################

        transform = LineGraph()
        line_graph_data = transform(concatenated_knn_data)
        z = line_graph_data.x
        z_index = line_graph_data.edge_index


        z = self.conv1(z, z_index).to('cuda') #can be res
        z = self.conv2(z, z_index).to('cuda')  #res
        z = z.reshape((z.shape[0] // self.num_nodes, -1))  # z=[16,1600]

        out1 = self.vbmlp(z)
        out = self.vbllayer(out1)
        categorical_obj = out.predictive
        loss_set=out.train_loss_fn

        out = categorical_obj.probs
        #print(out)
        #out = self.fcn(z)

        return out,loss_set,out1







