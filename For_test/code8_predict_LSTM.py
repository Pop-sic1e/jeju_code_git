# -*- coding: utf-8 -*-
"""
Purpose:
    This script generates validation route predictions using trained Hybrid GCN-LSTM
    route prediction models. For each route split size k = 8, 12, and 16, the script
    loads the corresponding validation OD segments and trained model checkpoint,
    performs adjacency-constrained beam search decoding, and merges segment-level
    predictions into traveler-level route sequences.

Input:
    - Hexagon road network graph:
      ./data/new_hexagraph/hexa_network_with_road.gpickle
    - Validation route segment files:
      ./data/sample_processed_inputs/Validation/shortcut_route/route_split_{k}/
    - Validation traveler attribute files with OD nodes:
      ./data/sample_processed_inputs/Validation/shortcut_route/traveler_proper_OD_{k}.csv
    - Trained model checkpoints:
      ./LSTM_weight/model_{k}.pth

Output:
    - LSTM-based traveler-level route predictions:
      ./expected_outputs/code8_prediction_LSTM/LSTM_{k}_by1.csv
    - Runtime log:
      ./expected_outputs/code8_prediction_LSTM/timing_log_adj_multi_k.txt

Main procedures:
    1. Load the hexagon road network graph and convert it into PyTorch Geometric edge_index format.
    2. Build an adjacency list for graph-constrained route decoding.
    3. Load validation OD route segments and traveler-level features for each k.
    4. Load the trained Hybrid GCN-LSTM checkpoint corresponding to each k.
    5. Decode route sequences using adjacency-constrained beam search.
    6. Merge segment-level predictions into traveler-level route sequences.
    7. Save predicted routes and runtime logs.
"""

import os
import re
import math
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv

import time
import multiprocessing as mp
from datetime import timedelta, datetime


# 0. Configuration
GRAPH_PATH  = "./data/new_hexagraph/hexa_network_with_road.gpickle"
VAL_ROOT    = "./data/sample_processed_inputs/Validation/shortcut_route"

# weight for k=8/12/16 should be placed at ./LSTM_weight/model_{k}.pth
WEIGHT_TMPL = "./LSTM_weight/model_{k}.pth"

OUT_DIR     = "./expected_outputs/code8_prediction_LSTM"
os.makedirs(OUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUT_DIR, "timing_log_adj_multi_k.txt")

device = torch.device("cpu")

#  8 worker × 3 threads per worker = 24 threads total
EVAL_WORKERS = int(os.environ.get("EVAL_WORKERS", "8"))
THREADS_PER_WORKER = int(os.environ.get("THREADS_PER_WORKER", "3"))

seed = 526
torch.manual_seed(seed)
np.random.seed(seed)


BEAM_SIZE = 5
MAX_DECODE_LEN_FACTOR = 1.2
MAX_DECODE_EXTRA = 50
REVISIT_PENALTY = 0.3
LENGTH_NORM_ALPHA = 0.7


# utility functions
def _fmt(sec: float) -> str:
    return str(timedelta(seconds=sec))

def _log_write(log_path: str, lines: list):
    text = "\n".join(lines) + "\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text)
    print(text, end="")

def numerical_sort_key(file):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", file)]

def chunk_indices(n, n_chunks):
    base = n // n_chunks
    rem = n % n_chunks
    chunks, start = [], 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        if size > 0:
            chunks.append((start, start + size))
            start += size
    return chunks

def build_adj_list(edge_index_cpu, num_nodes):
    src = edge_index_cpu[0].tolist()
    dst = edge_index_cpu[1].tolist()
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        adj[u].append(v)
        adj[v].append(u)

    out = []
    for nbrs in adj:
        if len(nbrs) == 0:
            out.append(torch.empty(0, dtype=torch.long))
        else:
            uniq = sorted(set(nbrs))
            out.append(torch.tensor(uniq, dtype=torch.long))
    return out


# 1. Model definition with an embedding-based graph encoder
class UserEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, p_drop=0.0):
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
    def __init__(self, num_nodes, node_emb_dim=64, out_dim=64, p_drop=0.0):
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
    def __init__(self, user_input_dim, embed_dim, hidden_dim, num_nodes,
                 gnn_out_dim=64, node_emb_dim=64, p_drop=0.0):
        super().__init__()
        self.user_encoder = UserEncoder(user_input_dim, embed_dim, p_drop=p_drop)
        self.graph_encoder = GraphEncoder(num_nodes, node_emb_dim=node_emb_dim, out_dim=gnn_out_dim, p_drop=p_drop)

        lstm_in = embed_dim + 2 * gnn_out_dim
        self.lstm = nn.LSTM(input_size=lstm_in, hidden_size=hidden_dim, batch_first=True)
        self.drop = nn.Dropout(p_drop)
        self.output = nn.Linear(hidden_dim, num_nodes)

    @torch.no_grad()
    def forward_next(self, user_feat, prefix_seq, end_node, graph_emb):
        """
        prefix_seq: [1, T]
        end_node:  [1]
        return: logits_next [1, V]
        """
        B, T = prefix_seq.shape

        idx = prefix_seq.clone()
        pad_mask = (idx == -1)
        idx[pad_mask] = 0

        cell_emb = graph_emb[idx]  # [1,T,Dg]
        if pad_mask.any():
            cell_emb[pad_mask] = 0.0

        user_emb = self.user_encoder(user_feat).unsqueeze(1).repeat(1, T, 1)
        end_emb = graph_emb[end_node]  # [1,Dg]

        step_w = torch.arange(1, T + 1, device=prefix_seq.device, dtype=torch.float32) / max(T, 1)
        end_emb_weighted = end_emb.unsqueeze(1) * step_w.unsqueeze(0).unsqueeze(-1)

        x = torch.cat([cell_emb, user_emb, end_emb_weighted], dim=-1)
        h, _ = self.lstm(x)
        h_last = self.drop(h[:, -1])
        logits = self.output(h_last)  # [1,V]
        return logits


# 2. Adjacency-constrained beam search decoding
@torch.no_grad()
def constrained_beam_search_decode(model, user_i, start_node, end_node, graph_emb, adj_list,
                                   beam_size=5, max_len=100,
                                   length_norm_alpha=0.7, revisit_penalty=0.3):
    # Candidate next nodes are restricted to the neighbors of the current node.
    # Length normalization and revisit penalty are applied during beam search.
    start_node = int(start_node)
    end_node = int(end_node)

    beams = [([start_node], 0.0)]   # (seq, logp_sum)
    finished = []

    def norm_score(seq, sc):
        return sc / (len(seq) ** length_norm_alpha)

    while True:
        new_beams = []
        all_done = True

        for seq, score in beams:
            last = seq[-1]

            # End condition: max length reached or end node generated
            if len(seq) >= max_len or last == end_node:
                finished.append((seq, score))
                continue

            nbrs = adj_list[last]
            if nbrs.numel() == 0:
                finished.append((seq, score))
                continue

            all_done = False

            prefix = torch.tensor([seq], dtype=torch.long, device=device)  # [1,T]
            end_t = torch.tensor([end_node], dtype=torch.long, device=device)

            logits = model.forward_next(user_i, prefix, end_t, graph_emb=graph_emb)  # [1,V]
            logp = F.log_softmax(logits, dim=-1).squeeze(0)  # [V]

            cand = nbrs.to(device)
            cand_logp = logp.index_select(0, cand)

            # penalty for revisiting already visited nodes in the current sequence
            if revisit_penalty and revisit_penalty > 0:
                visited = set(seq)
                penalty = torch.tensor(
                    [1.0 if int(v) in visited else 0.0 for v in cand.tolist()],
                    dtype=torch.float32, device=device
                )
                cand_logp = cand_logp - revisit_penalty * penalty

            k2 = min(beam_size, cand_logp.numel())
            topk = torch.topk(cand_logp, k=k2)

            for local_i, lp in zip(topk.indices.tolist(), topk.values.tolist()):
                nxt = int(cand[local_i].item())
                new_beams.append((seq + [nxt], score + float(lp)))

        if all_done:
            break

        new_beams.sort(key=lambda x: norm_score(x[0], x[1]), reverse=True)
        beams = new_beams[:beam_size]

    finished.extend(beams)
    best = max(finished, key=lambda x: norm_score(x[0], x[1]))
    return best[0]


# 3. Load validation data for a given k
def load_val_split(k):
    """
    shortcut_route/route_split_{k} + traveler_proper_OD_{k}.csv
    """
    csv_dir_Va = os.path.join(VAL_ROOT, f"route_split_{k}")
    files = sorted([f for f in os.listdir(csv_dir_Va) if f.endswith(".csv")], key=numerical_sort_key)

    files_iter = tqdm(files, desc=f"load files k={k}", unit="file")
    user_df = pd.read_csv(os.path.join(VAL_ROOT, f"traveler_proper_OD_{k}.csv"))
    user_feats_va = torch.tensor(user_df.iloc[:, 1:-2].values, dtype=torch.float32)

    travel_name_list, route_seq_list = [], []
    start_nodes, end_nodes = [], []

    for fn in files_iter:
        tmp = pd.read_csv(os.path.join(csv_dir_Va, fn))
        codes_int = tmp.iloc[:, 0].astype(int).tolist()
        seq = torch.tensor(codes_int, dtype=torch.long)

        route_seq_list.append(seq)
        travel_name_list.append(tmp.iloc[0, 1])
        start_nodes.append(int(seq[0].item()))
        end_nodes.append(int(seq[-1].item()))

    T_max = max(len(s) for s in route_seq_list)
    padded = torch.full((len(route_seq_list), T_max), fill_value=-1, dtype=torch.long)
    for i, s in enumerate(route_seq_list):
        padded[i, :len(s)] = s

    start_nodes = torch.tensor(start_nodes, dtype=torch.long)
    end_nodes = torch.tensor(end_nodes, dtype=torch.long)

    return user_feats_va, padded, start_nodes, end_nodes, travel_name_list


# 4. Merge segment-level predictions by traveler name
def merge_segments_by_name(names, paths_int):
    df = pd.DataFrame({"name": names, "path": paths_int})
    merged = {}
    for name, grp in df.groupby("name", sort=False):
        seqs = grp["path"].tolist()
        cleaned = []
        for i, cur in enumerate(seqs):
            if i < len(seqs) - 1:
                nxt = seqs[i + 1]
                if cur and nxt and cur[-1] == nxt[0]:
                    cur = cur[:-1]
            cleaned.append(cur)
        merged[name] = sum(cleaned, [])
    return pd.DataFrame({"name": list(merged.keys()), "path": list(merged.values())})


# 5. Global variables shared across forked worker processes
G_NUM_NODES = None
G_EDGE_INDEX = None
G_ADJ_LIST = None

G_STATE_DICT = None
G_USER_DIM = None

VAL_USER_FEATS = None
VAL_PADDED = None
VAL_START = None
VAL_END = None


# 6. Worker function for chunk-level decoding
def worker_decode_chunk(idx_start, idx_end, beam_size):
    # 3 threads per worker for intra-op parallelism
    os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_WORKER)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    torch.set_num_threads(THREADS_PER_WORKER)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    embed_dim = 64
    hidden_dim = 128
    gnn_out_dim = 64
    node_emb_dim = 64

    model = HybridRouteModel(
        user_input_dim=G_USER_DIM,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_nodes=G_NUM_NODES,
        gnn_out_dim=gnn_out_dim,
        node_emb_dim=node_emb_dim,
        p_drop=0.0
    ).to(device)

    model.load_state_dict(G_STATE_DICT, strict=True)
    model.eval()

    with torch.no_grad():
        graph_emb = model.graph_encoder(G_EDGE_INDEX)  # [N,Dg]
        preds = []

        for i in range(idx_start, idx_end):
            gt_full = VAL_PADDED[i].tolist()
            gt_seq = [x for x in gt_full if x != -1]
            if not gt_seq:
                preds.append([])
                continue

            user_i = VAL_USER_FEATS[i:i+1].to(device)  # [1,Du]
            end_i  = VAL_END[i].item()
            start_node = int(VAL_START[i].item())

            ml = int(math.ceil(len(gt_seq) * MAX_DECODE_LEN_FACTOR))
            ml = max(2, ml) + MAX_DECODE_EXTRA

            pred_seq = constrained_beam_search_decode(
                model=model,
                user_i=user_i,
                start_node=start_node,
                end_node=end_i,
                graph_emb=graph_emb,
                adj_list=G_ADJ_LIST,
                beam_size=beam_size,
                max_len=ml,
                length_norm_alpha=LENGTH_NORM_ALPHA,
                revisit_penalty=REVISIT_PENALTY
            )
            preds.append(pred_seq)

    return (idx_start, idx_end, preds)


# 7. Run prediction for each route split size
def run_prediction_for_k(k, beam_size=BEAM_SIZE):
    t_start_k = time.perf_counter()

    # load validation data for this k
    user_feats_va, padded_va, start_nodes, end_nodes, travel_names = load_val_split(k)
    N = padded_va.size(0)

    # load model weights for this k
    weight_path = WEIGHT_TMPL.format(k=k)
    if not os.path.isfile(weight_path):
        _log_write(LOG_PATH, [
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] k={k} - SKIP (weight not found)",
            f"  missing: {weight_path}",
            "-" * 60
        ])
        return

    state_dict = torch.load(weight_path, map_location="cpu")

    global G_STATE_DICT, G_USER_DIM, VAL_USER_FEATS, VAL_PADDED, VAL_START, VAL_END
    G_STATE_DICT = state_dict
    G_USER_DIM = user_feats_va.size(1)

    VAL_USER_FEATS = user_feats_va
    VAL_PADDED = padded_va
    VAL_START = start_nodes
    VAL_END = end_nodes

    n_workers = min(EVAL_WORKERS, max(1, N))
    chunks = chunk_indices(N, n_workers)

    t_decode_start = time.perf_counter()

    ctx = mp.get_context("fork")
    results = []

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = [ex.submit(worker_decode_chunk, s, e, beam_size) for (s, e) in chunks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"decode k={k}", unit="chunk"):
            results.append(fut.result())

    t_decode_end = time.perf_counter()

    # merge chunk-level predictions back to the original order and then by traveler name
    preds_int = [None] * N
    for s, e, part in results:
        for j, seq in enumerate(part):
            preds_int[s + j] = seq

    merged_df = merge_segments_by_name(travel_names, preds_int)

    out_path = os.path.join(OUT_DIR, f"LSTM_{k}_by1.csv")
    merged_df.to_csv(out_path, index=False)

    t_end_k = time.perf_counter()

    _log_write(LOG_PATH, [
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] k={k}",
        f"  weight        : {weight_path}",
        f"  saved file    : {out_path}",
        f"  samples       : {N}",
        f"  workers       : {n_workers}",
        f"  threads/worker: {THREADS_PER_WORKER}",
        f"  beam_size     : {beam_size}",
        f"  decode time   : {_fmt(t_decode_end - t_decode_start)}",
        f"  total time    : {_fmt(t_end_k - t_start_k)}",
        "-" * 60
    ])


# 8) main
if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    _log_write(LOG_PATH, [
        "=" * 60,
        f"Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"CPU plan: workers={EVAL_WORKERS} * threads_per_worker={THREADS_PER_WORKER} = {EVAL_WORKERS * THREADS_PER_WORKER}",
        "=" * 60
    ])

    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    data = from_networkx(G)
    num_nodes = data.num_nodes
    edge_index_cpu = data.edge_index.to(torch.long).to(device)

    # make adjacency list for constrained decoding
    adj_list = build_adj_list(edge_index_cpu, num_nodes)

    # set global graph variables for worker processes
    G_NUM_NODES = num_nodes
    G_EDGE_INDEX = edge_index_cpu
    G_ADJ_LIST = adj_list

    all_start = time.perf_counter()

    for k in tqdm([8, 12, 16], desc="all k", unit="set"):
        run_prediction_for_k(k, beam_size=BEAM_SIZE)

    all_end = time.perf_counter()

    _log_write(LOG_PATH, [
        f"ALL SETS TOTAL ELAPSED: {_fmt(all_end - all_start)}",
        f"Run finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60
    ])
