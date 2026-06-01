# -*- coding: utf-8 -*-
"""
Purpose:
    This script monitors newly saved Hybrid GCN-LSTM model checkpoints and evaluates
    them on the validation route segments using CPU-based parallel decoding. For each
    selected checkpoint, the script generates adjacency-constrained beam-search route
    predictions and records validation metrics.

Input:
    - Model checkpoints saved by the training script:
      ../weight_for_test/weight_8/model_epoch_{epoch}.pth
    - Hexagon road network graph:
      ../new_hexagraph/hexa_network_with_road.gpickle
    - Validation route segment files:
      ../Jeju_data/Validation/shortcut_route/route_split_8/
    - Validation traveler attribute file with OD nodes:
      ../Jeju_data/Validation/shortcut_route/traveler_proper_OD_8.csv

Output:
    - Validation metric log:
      ./LSTM_Result/eval_metrics_8.csv

Main procedures:
    1. Load the hexagon road network graph and validation route segment data.
    2. Monitor WEIGHT_DIR for newly saved model checkpoints.
    3. Evaluate checkpoints at the specified epoch interval.
    4. Generate validation routes using adjacency-constrained beam search.
    5. Compute Jaccard similarity, Levenshtein distance, and end-node reach rate.
    6. Save validation metrics to a CSV file.
    7. Resume evaluation from the latest recorded epoch if the script is restarted.

Notes:
    1. Run this validation/evaluation monitoring script first.
    2. Run the GPU training script after this script has started.
    3. During training, newly saved checkpoints are automatically detected and evaluated.
"""

import os
import re
import time
import csv
import math
import pickle
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx


# setup
cpu_n = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
device = torch.device("cpu")

WEIGHT_DIR = "../weight_for_test/weight_8/"
CKPT_PATTERN = r"model_epoch_(\d+)\.pth"
POLL_SEC = 5

GRAPH_PATH = "../new_hexagraph/hexa_network_with_road.gpickle"

VAL_ROUTE_DIR = "../Jeju_data/Validation/shortcut_route/route_split_8/"
VAL_PROP_CSV  = "../Jeju_data/Validation/shortcut_route/traveler_proper_OD_8.csv"

OUT_DIR = "./LSTM_Result"
os.makedirs(OUT_DIR, exist_ok=True)
METRIC_CSV = os.path.join(OUT_DIR, "eval_metrics_8.csv")

# hyperparameters for evaluation
EVAL_EVERY_N_EPOCH = 50

BEAM_SIZE = 5
MAX_DECODE_LEN_FACTOR = 1.2
MAX_DECODE_EXTRA = 50
REVISIT_PENALTY = 0.3
LENGTH_NORM_ALPHA = 0.7

EVAL_WORKERS = int(os.environ.get("EVAL_WORKERS", "8"))
THREADS_PER_WORKER = int(os.environ.get("THREADS_PER_WORKER", "2"))

MAX_EVAL_SAMPLES = os.environ.get("MAX_EVAL_SAMPLES", "")
MAX_EVAL_SAMPLES = int(MAX_EVAL_SAMPLES) if str(MAX_EVAL_SAMPLES).strip() else None


# ============================================================
# Model
# ============================================================
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


# ============================================================
# Utility functions
# ============================================================
def numerical_sort_key(file):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", file)]

def jaccard_similarity(set_a, set_b):
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0

def levenshtein(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]

def ensure_csv_header(path: str):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "jaccard", "levenshtein", "end_reach", "n_samples", "ckpt"])

def append_csv_row(path: str, row: dict):
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([row["epoch"], row["jaccard"], row["levenshtein"], row["end_reach"], row["n_samples"], row["ckpt"]])

def extract_epoch(fn: str):
    m = re.search(CKPT_PATTERN, fn)
    return int(m.group(1)) if m else None

def list_ckpts():
    if not os.path.isdir(WEIGHT_DIR):
        return []
    out = []
    for fn in os.listdir(WEIGHT_DIR):
        ep = extract_epoch(fn)
        if ep is not None:
            out.append((ep, os.path.join(WEIGHT_DIR, fn)))
    out.sort(key=lambda x: x[0])
    return out

def should_eval_epoch(ep: int):
    return (ep % int(EVAL_EVERY_N_EPOCH)) == 0

def wait_file_ready(path: str, tries: int = 6, sleep_sec: float = 0.2) -> bool:
    last = None
    for _ in range(tries):
        try:
            sz = os.path.getsize(path)
        except OSError:
            time.sleep(sleep_sec)
            continue
        if last is not None and sz == last:
            return True
        last = sz
        time.sleep(sleep_sec)
    return True

def build_adj_list(edge_index_cpu, num_nodes):
    src = edge_index_cpu[0].tolist()
    dst = edge_index_cpu[1].tolist()
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        adj[u].append(v)
        adj[v].append(u)
    # 중복 제거
    out = []
    for nbrs in adj:
        if len(nbrs) == 0:
            out.append(torch.empty(0, dtype=torch.long))
        else:
            uniq = sorted(set(nbrs))
            out.append(torch.tensor(uniq, dtype=torch.long))
    return out

def load_resume_from_metric_csv(metric_csv_path: str):
    """
    Read METRIC_CSV and return:
    - done_epochs: set of epochs that have already been evaluated
    - resume_from_epoch: maximum recorded epoch + 1
    """
    done = set()
    last_ep = None

    if (not os.path.exists(metric_csv_path)) or (os.path.getsize(metric_csv_path) == 0):
        return done, 0

    try:
        with open(metric_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames or "epoch" not in reader.fieldnames:
                return done, 0

            for row in reader:
                try:
                    ep = int(str(row.get("epoch", "")).strip())
                except Exception:
                    continue
                done.add(ep)
                if last_ep is None or ep > last_ep:
                    last_ep = ep

    except Exception as e:
        print(f"[Eval] warning: failed to read {metric_csv_path}: {repr(e)}")
        return set(), 0

    resume_from = (last_ep + 1) if last_ep is not None else 0
    return done, resume_from


@torch.no_grad()
def constrained_beam_search_decode(model, user_i, start_node, end_node, graph_emb, adj_list,
                                   beam_size=5, max_len=100,
                                   length_norm_alpha=0.7, revisit_penalty=0.3):
    start_node = int(start_node)
    end_node = int(end_node)

    beams = [([start_node], 0.0)]  # (seq, logp)
    finished = []

    def norm_score(seq, sc):
        return sc / (len(seq) ** length_norm_alpha)

    while True:
        new_beams = []
        all_done = True

        for seq, score in beams:
            last = seq[-1]

            if len(seq) >= max_len or last == end_node:
                finished.append((seq, score))
                continue

            nbrs = adj_list[last]
            if nbrs.numel() == 0:
                finished.append((seq, score))
                continue

            all_done = False

            prefix = torch.tensor([seq], dtype=torch.long, device=device)
            end_t = torch.tensor([end_node], dtype=torch.long, device=device)

            logits = model.forward_next(user_i, prefix, end_t, graph_emb=graph_emb)  # [1,V]
            logp = F.log_softmax(logits, dim=-1).squeeze(0)                          # [V]

            cand = nbrs
            cand_logp = logp.index_select(0, cand)

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
    return best[0], best[1]


# ============================================================
# Global Parameters
# ============================================================
G_NUM_NODES = None
G_EDGE_INDEX = None
G_ADJ_LIST = None

VAL_USER_FEATS = None
VAL_GT_LIST = None
VAL_END_LIST = None
VAL_MAXLEN_LIST = None
VAL_N = None

EMBED_DIM    = 64
HIDDEN_DIM   = 128
GNN_OUT_DIM  = 64
NODE_EMB_DIM = 64


# ============================================================
# Worker function for evaluating one checkpoint
# ============================================================
def eval_one_ckpt(ep: int, ckpt_path: str):
    os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_WORKER)
    torch.set_num_threads(THREADS_PER_WORKER)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    try:
        wait_file_ready(ckpt_path)

        model = HybridRouteModel(
            user_input_dim=VAL_USER_FEATS.size(1),
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_nodes=G_NUM_NODES,
            gnn_out_dim=GNN_OUT_DIM,
            node_emb_dim=NODE_EMB_DIM,
            p_drop=0.0
        ).to(device)

        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd)
        model.eval()

        with torch.no_grad():
            graph_emb = model.graph_encoder(G_EDGE_INDEX)  # [N,Dg]

            total_jac, total_lev = 0.0, 0.0
            total_reach, cnt = 0, 0

            n_eval = VAL_N
            if MAX_EVAL_SAMPLES is not None:
                n_eval = min(n_eval, MAX_EVAL_SAMPLES)

            for i in range(n_eval):
                gt = VAL_GT_LIST[i]
                if not gt:
                    continue

                user_i = VAL_USER_FEATS[i:i+1]  # [1,Du]
                end_i = VAL_END_LIST[i]
                start_node = gt[0]
                max_len = VAL_MAXLEN_LIST[i]

                pred_seq, _ = constrained_beam_search_decode(
                    model, user_i, start_node, end_i,
                    graph_emb, G_ADJ_LIST,
                    beam_size=BEAM_SIZE,
                    max_len=max_len,
                    length_norm_alpha=LENGTH_NORM_ALPHA,
                    revisit_penalty=REVISIT_PENALTY
                )

                total_jac += jaccard_similarity(set(gt), set(pred_seq))
                total_lev += levenshtein(gt, pred_seq)
                total_reach += int(len(pred_seq) > 0 and pred_seq[-1] == end_i)
                cnt += 1

        if cnt == 0:
            return {"ok": True, "epoch": ep, "cnt": 0, "ckpt": os.path.basename(ckpt_path)}

        return {
            "ok": True,
            "epoch": ep,
            "jaccard": total_jac / cnt,
            "levenshtein": total_lev / cnt,
            "end_reach": total_reach / cnt,
            "n_samples": cnt,
            "ckpt": os.path.basename(ckpt_path),
        }

    except Exception as e:
        return {
            "ok": False,
            "epoch": ep,
            "ckpt": os.path.basename(ckpt_path),
            "error": repr(e),
            "trace": traceback.format_exc(limit=8)
        }


# ============================================================
# main
# ============================================================
def main():
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    os.environ["OMP_NUM_THREADS"] = str(cpu_n)
    os.environ["MKL_NUM_THREADS"] = str(cpu_n)
    torch.set_num_threads(cpu_n)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # graph
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    data = from_networkx(G)
    num_nodes = data.num_nodes
    edge_index = data.edge_index.to(device)

    adj_list = build_adj_list(edge_index, num_nodes)

    print(f"[Graph] num_nodes={num_nodes:,} | num_edges={edge_index.size(1):,}")

    # load validation data
    route_list = sorted([f for f in os.listdir(VAL_ROUTE_DIR) if f.endswith(".csv")], key=numerical_sort_key)
    user_feature_va = pd.read_csv(VAL_PROP_CSV)
    user_feats_va = torch.tensor(user_feature_va.iloc[:, 1:-2].values, dtype=torch.float32, device=device)

    gt_list = []
    end_list = []
    maxlen_list = []

    for fn in route_list:
        tmp = pd.read_csv(os.path.join(VAL_ROUTE_DIR, fn))
        seq = tmp.iloc[:, 0].values.tolist()
        seq = [int(v) for v in seq if str(v) != "nan"]
        if len(seq) < 2:
            gt_list.append([])
            end_list.append(0)
            maxlen_list.append(2)
            continue

        gt_list.append(seq)
        end_list.append(int(seq[-1]))

        # decode max_len
        ml = int(math.ceil(len(seq) * MAX_DECODE_LEN_FACTOR))
        ml = max(2, ml) + MAX_DECODE_EXTRA
        maxlen_list.append(ml)

    n_val = len(gt_list)
    print(f"[Eval] validation loaded: n={n_val:,} | feat_dim={user_feats_va.size(1)}")
    print(f"[Eval] workers={EVAL_WORKERS} | threads_per_worker={THREADS_PER_WORKER} | poll={POLL_SEC}s")
    if MAX_EVAL_SAMPLES is not None:
        print(f"[Eval] MAX_EVAL_SAMPLES={MAX_EVAL_SAMPLES}")

    # param setting
    global G_NUM_NODES, G_EDGE_INDEX, G_ADJ_LIST
    global VAL_USER_FEATS, VAL_GT_LIST, VAL_END_LIST, VAL_MAXLEN_LIST, VAL_N

    G_NUM_NODES = num_nodes
    G_EDGE_INDEX = edge_index
    G_ADJ_LIST = adj_list

    VAL_USER_FEATS = user_feats_va
    VAL_GT_LIST = gt_list
    VAL_END_LIST = end_list
    VAL_MAXLEN_LIST = maxlen_list
    VAL_N = n_val

    # run evaluation
    ensure_csv_header(METRIC_CSV)
    done_epochs, resume_from_epoch = load_resume_from_metric_csv(METRIC_CSV)
    print(f"[Eval] resume_from_epoch={resume_from_epoch} | already_done={len(done_epochs)}")

    inflight_epochs = set()
    inflight = {}

    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=EVAL_WORKERS, mp_context=ctx) as ex:
        print(f"[Eval] watching: {WEIGHT_DIR} | eval_every={EVAL_EVERY_N_EPOCH}")

        while True:
            # new ckpt detection
            ckpts = list_ckpts()
            for ep, ckpt_path in ckpts:
                # epoch selection
                if ep < resume_from_epoch:
                    continue

                if ep in done_epochs:
                    continue
                if ep in inflight_epochs:
                    continue
                if not should_eval_epoch(ep):
                    continue
                if not os.path.isfile(ckpt_path):
                    continue

                wait_file_ready(ckpt_path)
                fut = ex.submit(eval_one_ckpt, ep, ckpt_path)
                inflight[fut] = (ep, ckpt_path)
                inflight_epochs.add(ep)
                print(f"[Eval] submitted epoch={ep} ({os.path.basename(ckpt_path)}) | inflight={len(inflight)}")

            # result collection
            if inflight:
                done_set, _ = wait(list(inflight.keys()), timeout=POLL_SEC, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    ep, ckpt_path = inflight.pop(fut)
                    inflight_epochs.discard(ep)

                    try:
                        res = fut.result()
                    except Exception as e:
                        print(f"[Eval] epoch={ep} | worker crashed: {repr(e)}")
                        continue

                    if not res.get("ok", False):
                        print(f"[Eval] epoch={ep} | failed | {res.get('error')}\n{res.get('trace','')}")
                        continue

                    if res.get("cnt", res.get("n_samples", 0)) == 0:
                        print(f"[Eval] epoch={ep} | no valid samples")
                        done_epochs.add(ep)
                        resume_from_epoch = max(resume_from_epoch, ep + 1)
                        continue

                    print(
                        f"[Eval] epoch={ep} | "
                        f"Jaccard={res['jaccard']:.4f} | "
                        f"Lev={res['levenshtein']:.2f} | "
                        f"EndReach={res['end_reach']:.3f} | "
                        f"n={res['n_samples']}"
                    )
                    append_csv_row(METRIC_CSV, res)
                    done_epochs.add(ep)
                    resume_from_epoch = max(resume_from_epoch, ep + 1)

            else:
                time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
