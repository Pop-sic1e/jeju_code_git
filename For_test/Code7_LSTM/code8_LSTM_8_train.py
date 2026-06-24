# -*- coding: utf-8 -*-
"""
Purpose:
    This script trains the Hybrid GCN-LSTM route prediction model on GPU and saves
    model checkpoints at regular intervals. The model uses graph-connected route
    segments, traveler-level attributes, and destination-node information to predict
    the next node in a route sequence.

Input:
    - Hexagon road network graph:
      ../data/new_hexagraph/hexa_network_with_road.gpickle
    - Training route segment files:
      ../data/sample_processed_inputs/Training/shortcut_route/route_split_8/
    - Traveler attribute file with OD nodes:
      ../data/sample_processed_inputs/Training/shortcut_route/traveler_proper_OD_8.csv

Output:
    - Model checkpoints:
      ../expected_outputs/code7_weight_for_test/weight_8/model_epoch_{epoch}.pth

Main procedures:
    1. Load the hexagon road network graph and convert it into PyTorch Geometric
       edge_index format.
    2. Load graph-connected training route segments and traveler-level feature vectors.
    3. Pad variable-length route sequences using -1 as the padding value.
    4. Train the Hybrid GCN-LSTM model using teacher forcing over route sequences.
    5. Add an auxiliary destination-node prediction loss.
    6. Save model checkpoints using atomic file replacement to prevent partially
       written checkpoint files.

Notes:
    - This script only performs GPU-based model training and checkpoint saving.
      It does not perform validation, beam-search decoding, or CPU evaluation.
    - In the actual execution workflow, run the validation/evaluation monitoring
      script first, and then run this training script. The validation script monitors
      newly saved checkpoints and evaluates them as training progresses.
"""

import os
import re
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Environment and hyperparameter settings
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("GPU:", torch.cuda.get_device_name(0))

cpu_n = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
torch.set_num_threads(cpu_n)
try:
    torch.set_num_interop_threads(2)
except Exception:
    pass
os.environ["OMP_NUM_THREADS"] = str(cpu_n)
os.environ["MKL_NUM_THREADS"] = str(cpu_n)

seed = 526
torch.manual_seed(seed)
np.random.seed(seed)

# File paths
GRAPH_PATH = "../data/new_hexagraph/hexa_network_with_road.gpickle"

TRAIN_ROUTE_DIR = "../data/sample_processed_inputs/Training/shortcut_route/route_split_8/"
TRAIN_PROP_CSV  = "../data/sample_processed_inputs/Training/shortcut_route/traveler_proper_OD_8.csv"

WEIGHT_DIR = "../expected_outputs/code7_weight_for_test/weight_8/"
os.makedirs(WEIGHT_DIR, exist_ok=True)

# Training hyperparameters
batch_size    = 8192
num_epochs    = 30000
lr            = 1e-4
weight_decay  = 1e-4
save_interval = 10     # ckpt save step
print_interval = 10    # training log print step

# Model hyperparameters
EMBED_DIM    = 64
HIDDEN_DIM   = 128
GNN_OUT_DIM  = 64
NODE_EMB_DIM = 64
P_DROP       = 0.2


# utility functions
def numerical_sort_key(file):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", file)]

def atomic_torch_save(state_dict, path: str):
    tmp_path = path + ".tmp"
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, path)


# =========================
# Model
# =========================
class UserEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(16, embed_dim)
        )

    def forward(self, user_feat):
        return self.net(user_feat)

class GraphEncoder(nn.Module):
    # Trainable node embedding followed by two GCNConv layers.
    def __init__(self, num_nodes, node_emb_dim=64, out_dim=64, p_drop=0.2):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, node_emb_dim)
        self.conv1 = GCNConv(node_emb_dim, out_dim)
        self.conv2 = GCNConv(out_dim, out_dim)
        self.drop = nn.Dropout(p_drop)

        nn.init.normal_(self.node_emb.weight, mean=0.0, std=0.02)

    def forward(self, edge_index):
        x = self.node_emb.weight
        x = self.drop(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.drop(x)
        x = self.conv2(x, edge_index)
        return x  # [N, out_dim]

class HybridRouteModel(nn.Module):
    """
    Hybrid route prediction model with safe handling of padding values.

    Padding values are set to -1. Before indexing node embeddings, -1 values are
    temporarily replaced with 0, and the corresponding embeddings are masked out.
    """
    def __init__(self, user_input_dim, embed_dim, hidden_dim, num_nodes,
                 gnn_out_dim=64, node_emb_dim=64, p_drop=0.2):
        super().__init__()
        self.user_encoder = UserEncoder(user_input_dim, embed_dim, p_drop=p_drop)
        self.graph_encoder = GraphEncoder(num_nodes, node_emb_dim=node_emb_dim, out_dim=gnn_out_dim, p_drop=p_drop)

        lstm_in = embed_dim + 2 * gnn_out_dim
        self.lstm = nn.LSTM(input_size=lstm_in, hidden_size=hidden_dim, batch_first=True)
        self.drop = nn.Dropout(p_drop)
        self.output = nn.Linear(hidden_dim, num_nodes)

    def forward(self, user_feat, partial_cell_seq, end_node,
                edge_index=None, graph_emb=None):
        """
        user_feat: [B, Du]
        partial_cell_seq: [B, T]
        end_node: [B]
        """
        B, T = partial_cell_seq.shape

        if graph_emb is None:
            assert edge_index is not None
            graph_emb = self.graph_encoder(edge_index)  # [N, Dg]

        idx = partial_cell_seq.clone()
        pad_mask = (idx == -1)
        idx[pad_mask] = 0

        cell_emb = graph_emb[idx]  # [B, T, Dg]
        if pad_mask.any():
            cell_emb[pad_mask] = 0.0

        user_emb = self.user_encoder(user_feat).unsqueeze(1).repeat(1, T, 1)  # [B, T, Du]
        end_emb = graph_emb[end_node]                                         # [B, Dg]

        # Apply time-dependent destination-node weighting.
        step_w = torch.arange(1, T + 1, device=partial_cell_seq.device, dtype=torch.float32) / max(T, 1)
        end_emb_weighted = end_emb.unsqueeze(1) * step_w.unsqueeze(0).unsqueeze(-1)  # [B, T, Dg]

        x = torch.cat([cell_emb, user_emb, end_emb_weighted], dim=-1)
        h, _ = self.lstm(x)
        h_last = self.drop(h[:, -1])
        logits = self.output(h_last)  # [B, V]
        return logits


# =========================
# Load data for training
# =========================
with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)

data = from_networkx(G)
num_nodes = data.num_nodes
edge_index = data.edge_index.to(device)

print(f"[Graph] num_nodes={num_nodes:,} | num_edges={edge_index.size(1):,}")


# Dataset and collate function
class RouteDataset(Dataset):
    def __init__(self, seq_list, user_feats):
        assert len(seq_list) == user_feats.size(0), "not matching number of sequences and user features"
        self.seq_list = seq_list
        self.user_feats = user_feats

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, idx):
        seq = self.seq_list[idx]
        end_node = seq[-1].item()
        user = self.user_feats[idx]
        return seq, user, end_node

def collate_pad(batch, padding_value=-1):
    seqs, users, ends = zip(*batch)
    padded_seq = pad_sequence(seqs, batch_first=True, padding_value=padding_value)  # [B, T]
    user_batch = torch.stack(users, dim=0)                                          # [B, Du]
    end_nodes = torch.tensor(ends, dtype=torch.long)                                # [B]
    return padded_seq, user_batch, end_nodes


# Data loading (Train)
route_split_list = sorted(
    [f for f in os.listdir(TRAIN_ROUTE_DIR) if f.endswith(".csv")],
    key=numerical_sort_key
)

user_feature = pd.read_csv(TRAIN_PROP_CSV)
user_feats = torch.tensor(user_feature.iloc[:, 1:-2].values, dtype=torch.float32)

route_seq_list = []
for fn in route_split_list:
    tmp = pd.read_csv(os.path.join(TRAIN_ROUTE_DIR, fn))
    seq = torch.tensor(tmp.iloc[:, 0].values, dtype=torch.long)
    route_seq_list.append(seq)

train_ds = RouteDataset(route_seq_list, user_feats)
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda b: collate_pad(b, padding_value=-1),
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(f"[Data] Train samples={len(train_ds):,} | feat_dim={user_feats.size(1)}")


# =========================
# Start training
# =========================
model = HybridRouteModel(
    user_input_dim=user_feats.size(1),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_nodes=num_nodes,
    gnn_out_dim=GNN_OUT_DIM,
    node_emb_dim=NODE_EMB_DIM,
    p_drop=P_DROP
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# Optional: load an existing checkpoint when resuming GPU training.
# state = torch.load('./weight_8/model_epoch_60000.pth', map_location=device, weights_only=True)
# model.load_state_dict(state)
print("=== Load Weights (GPU only, save ckpt) ===")
print("=== Training Start (GPU only, save ckpt) ===")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_terms = 0

    for padded_seq, user_batch, end_nodes_batch in train_loader:
        # Recompute graph embeddings for each batch to allow gradient updates.
        graph_emb = model.graph_encoder(edge_index)

        padded_seq = padded_seq.to(device, non_blocking=True)
        user_batch = user_batch.to(device, non_blocking=True)
        end_nodes_batch = end_nodes_batch.to(device, non_blocking=True)

        B, T = padded_seq.shape
        total_loss = torch.tensor(0.0, device=device)
        total_terms = 0

        # Apply teacher forcing over sequence steps.
        for t in range(1, T):
            partial_seq = padded_seq[:, :t]
            target = padded_seq[:, t]
            mask = (target != -1)
            if mask.any():
                logits = model(user_batch, partial_seq, end_nodes_batch, graph_emb=graph_emb)
                loss = F.cross_entropy(logits[mask], target[mask], reduction="mean")
                total_loss = total_loss + loss
                total_terms += 1

        # Add auxiliary destination-node prediction loss.
        if T > 1:
            logits_end = model(user_batch, padded_seq[:, :-1], end_nodes_batch, graph_emb=graph_emb)
            loss_end = F.cross_entropy(logits_end, end_nodes_batch, reduction="mean")
            total_loss = total_loss + loss_end
            total_terms += 1

        batch_loss = total_loss / max(total_terms, 1)

        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += float(batch_loss.detach().item())
        running_terms += 1

    avg_train_loss = running_loss / max(running_terms, 1)

    if (epoch + 1) % print_interval == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] | TrainLoss: {avg_train_loss:.4f}")

    if (epoch + 1) % save_interval == 0:
        ckpt_path = os.path.join(WEIGHT_DIR, f"model_epoch_{epoch+1}.pth")
        atomic_torch_save(model.state_dict(), ckpt_path)
