import torch
from src.models import BrainNN, SAGE
from torch_geometric.data import Data
from typing import List


def build_model(args, device, model_name, num_features, num_nodes):

    if model_name == 'sage':
        model = BrainNN(args,
                        SAGE(num_features, args, num_nodes,num_classes=2)
                        ).to(device)

    else:
        raise ValueError(f"ERROR: Model variant \"{args.variant}\" not found!")
    return model
