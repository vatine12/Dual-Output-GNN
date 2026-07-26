"""
fixed_ablation_common.py
Clean shared implementation module for trainable BART + R-GCN WebNLG notebooks.

Expected processed files under PROCESSED_DIR:
  - webnlg_processed.pkl
  - vocabularies.pkl
  - graphs_train.pkl
  - graphs_dev.pkl
  - graphs_test.pkl

This module supports:
  - pretrained BART loading with train_bart=True/False
  - Fusion-only model
  - Full dual-output model
  - optional constraint-only model
  - true-resume checkpoint training
  - generation evaluation with per-sample analysis
"""

import os
import json
import pickle
import random
import time
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn import RGCNConv

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from rouge_score import rouge_scorer


# ============================================================
# Reproducibility / paths / artifact loading
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_artifacts(processed_dir: str):
    data_path = os.path.join(processed_dir, "webnlg_processed.pkl")
    vocab_path = os.path.join(processed_dir, "vocabularies.pkl")

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Missing processed data: {data_path}")
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"Missing vocabulary file: {vocab_path}")

    with open(data_path, "rb") as f:
        data = pickle.load(f)
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    graphs = {}
    for split in ["train", "dev", "test"]:
        graph_path = os.path.join(processed_dir, f"graphs_{split}.pkl")
        if not os.path.isfile(graph_path):
            raise FileNotFoundError(f"Missing graph file: {graph_path}")
        with open(graph_path, "rb") as f:
            graphs[split] = pickle.load(f)

    return data, graphs, vocab


def infer_total_relations(graphs, vocab=None):
    """Infer num_relations from graph edge_type values; fallback to vocab."""
    max_rel = -1
    for graph_list in graphs.values():
        for g in graph_list:
            if hasattr(g, "edge_type") and g.edge_type is not None and g.edge_type.numel() > 0:
                max_rel = max(max_rel, int(g.edge_type.max().item()))
    if max_rel >= 0:
        return max_rel + 1

    if vocab is not None:
        if "total_relations" in vocab:
            return int(vocab["total_relations"])
        if "relation_vocab" in vocab:
            return len(vocab["relation_vocab"]) * 2
        if "relations" in vocab:
            return len(vocab["relations"]) * 2

    raise ValueError("Could not infer TOTAL_RELATIONS from graphs or vocab.")


def build_global_entity_set(data, vocab=None):
    if vocab is not None:
        if "global_entity_set" in vocab:
            return set(vocab["global_entity_set"])
        if "entity_vocab" in vocab:
            return set(vocab["entity_vocab"].keys())
        if "entities" in vocab:
            return set(vocab["entities"])

    ents = set()
    for split_data in data.values():
        for ex in split_data:
            for t in ex.get("triples", []):
                if isinstance(t, dict):
                    s = t.get("subject", "")
                    o = t.get("object", "")
                else:
                    s = t[0]
                    o = t[2]
                if str(s).strip():
                    ents.add(str(s).strip())
                if str(o).strip():
                    ents.add(str(o).strip())
    return ents


# ============================================================
# BART loading / optimizer helpers
# ============================================================

def load_pretrained_bart(model_name_or_path="facebook/bart-base", device="cuda", train_bart=True):
    from transformers import BartForConditionalGeneration, BartTokenizer

    tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
    bart = BartForConditionalGeneration.from_pretrained(model_name_or_path)

    for p in bart.parameters():
        p.requires_grad = bool(train_bart)

    bart = bart.to(device)
    bart.train() if train_bart else bart.eval()
    return tokenizer, bart


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def build_trainable_optimizer(model, bart_lr=3e-5, kg_lr=1e-4, weight_decay=0.01):
    bart_params = []
    kg_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("bart."):
            bart_params.append(p)
        else:
            kg_params.append(p)

    param_groups = []
    if bart_params:
        param_groups.append({"params": bart_params, "lr": bart_lr, "weight_decay": weight_decay})
    if kg_params:
        param_groups.append({"params": kg_params, "lr": kg_lr, "weight_decay": weight_decay})
    if not param_groups:
        raise ValueError("No trainable parameters found.")

    return torch.optim.AdamW(param_groups)


# ============================================================
# Graph encoder / fusion / constraint modules
# ============================================================

class DualOutputRGCN(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=384, output_dim=768,
                 num_relations=None, num_bases=30, dropout=0.1):
        super().__init__()
        if num_relations is None:
            raise ValueError("num_relations must be provided")
        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.fusion_head = nn.Linear(hidden_dim, output_dim)
        self.constraint_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type):
        h = self.conv1(x, edge_index, edge_type)
        h = self.dropout(self.relu(h))
        h = self.conv2(h, edge_index, edge_type)
        h = self.dropout(self.relu(h))
        h_kg_nodes = self.fusion_head(h)
        c_entity = self.constraint_head(h)
        return h_kg_nodes, c_entity


class KGCrossAttention(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

    def forward(self, h_lm, h_kg_padded, kg_mask=None):
        key_padding_mask = (kg_mask == 0) if kg_mask is not None else None
        h_kg_attn, attn_weights = self.cross_attn(
            h_lm, h_kg_padded, h_kg_padded, key_padding_mask=key_padding_mask
        )
        h_kg_attn = self.layer_norm(h_kg_attn)
        lam = self.gate(torch.cat([h_lm, h_kg_attn], dim=-1))
        h_fused = lam * h_lm + (1.0 - lam) * h_kg_attn
        return h_fused, lam, attn_weights


class TrieNode:
    def __init__(self):
        self.children = {}
        self.score = None


class TrieMap:
    def __init__(self, entity_names, entity_scores, tokenizer):
        self.root = TrieNode()
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.entity_names = entity_names
        for name, score in zip(entity_names, entity_scores):
            score = score.item() if hasattr(score, "item") else float(score)
            for surface in self._surface_forms(name):
                ids = tokenizer.encode(surface, add_special_tokens=False)
                if not ids:
                    continue
                node = self.root
                for tid in ids:
                    node = node.children.setdefault(int(tid), TrieNode())
                if node.score is None or abs(score) > abs(node.score):
                    node.score = score

    @staticmethod
    def _surface_forms(name):
        raw = str(name).strip()
        clean = raw.replace("_", " ").strip()
        forms = []
        for x in [raw, clean]:
            if x and x not in forms:
                forms.append(x)
        return forms

    def _subtree_score(self, node):
        if node.score is not None:
            return node.score
        if not node.children:
            return 0.0
        scores = [self._subtree_score(c) for c in node.children.values()]
        return float(sum(scores) / len(scores)) if scores else 0.0

    def get_constraint_vector(self, generated_ids):
        c_tok = torch.zeros(self.vocab_size, dtype=torch.float)

        # Bias entity-start tokens.
        for tid, child in self.root.children.items():
            c_tok[tid] += self._subtree_score(child)

        # Bias legal continuation if currently inside an entity prefix.
        if generated_ids:
            search_start = max(0, len(generated_ids) - 20)
            for start in range(search_start, len(generated_ids)):
                node = self.root
                ok = True
                for pos in range(start, len(generated_ids)):
                    tid = int(generated_ids[pos])
                    if tid not in node.children:
                        ok = False
                        break
                    node = node.children[tid]
                if ok and node.children:
                    for tid, child in node.children.items():
                        c_tok[tid] += 2.0 * self._subtree_score(child)
        return c_tok


def add_final_logits_bias(bart_model, logits):
    if hasattr(bart_model, "final_logits_bias") and bart_model.final_logits_bias is not None:
        return logits + bart_model.final_logits_bias.to(logits.device)
    return logits


def signed_entity_scores(c_entity):
    return 2.0 * torch.sigmoid(c_entity.squeeze(-1)) - 1.0


def pad_kg_nodes(h_kg_nodes, batch_vector, batch_size):
    hidden_dim = h_kg_nodes.size(1)
    nodes_per_graph = [(batch_vector == i).sum().item() for i in range(batch_size)]
    max_nodes = max(max(nodes_per_graph), 1)
    padded = torch.zeros(batch_size, max_nodes, hidden_dim, device=h_kg_nodes.device, dtype=h_kg_nodes.dtype)
    mask = torch.zeros(batch_size, max_nodes, device=h_kg_nodes.device, dtype=torch.float)
    offset = 0
    for i, n in enumerate(nodes_per_graph):
        if n > 0:
            padded[i, :n] = h_kg_nodes[offset:offset+n]
            mask[i, :n] = 1.0
            offset += n
    return padded, mask


# ============================================================
# Model variants
# ============================================================

class FusionOnlyGNNModel(nn.Module):
    VARIANT = "fusion_only"

    def __init__(self, bart_model, num_relations, gnn_hidden=384, num_bases=30):
        super().__init__()
        hidden_dim = bart_model.config.d_model
        self.bart = bart_model
        self.rgcn = DualOutputRGCN(hidden_dim, gnn_hidden, hidden_dim, num_relations, num_bases)
        self.kg_cross_attention = KGCrossAttention(hidden_dim)

    def forward(self, input_ids, attention_mask, decoder_input_ids,
                graph_x, graph_edge_index, graph_edge_type, graph_batch, labels=None):
        enc = self.bart.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_kg, c_entity = self.rgcn(graph_x, graph_edge_index, graph_edge_type)
        h_kg_pad, kg_mask = pad_kg_nodes(h_kg, graph_batch, input_ids.size(0))
        dec = self.bart.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=enc.last_hidden_state,
            encoder_attention_mask=attention_mask,
        )
        h_fused, gate_values, _ = self.kg_cross_attention(dec.last_hidden_state, h_kg_pad, kg_mask)
        logits = add_final_logits_bias(self.bart, self.bart.lm_head(h_fused))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return {
            "loss": loss,
            "logits": logits,
            "gate_values": gate_values,
            "c_entity": c_entity,
            "encoder_hidden_states": enc.last_hidden_state,
            "h_kg_nodes": h_kg,
        }


class ConstraintOnlyGNNModel(nn.Module):
    VARIANT = "constraint_only"

    def __init__(self, bart_model, num_relations, gnn_hidden=384, num_bases=30):
        super().__init__()
        hidden_dim = bart_model.config.d_model
        self.bart = bart_model
        self.rgcn = DualOutputRGCN(hidden_dim, gnn_hidden, hidden_dim, num_relations, num_bases)

    def forward(self, input_ids, attention_mask, decoder_input_ids,
                graph_x, graph_edge_index, graph_edge_type, graph_batch, labels=None):
        enc = self.bart.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_kg, c_entity = self.rgcn(graph_x, graph_edge_index, graph_edge_type)
        dec = self.bart.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=enc.last_hidden_state,
            encoder_attention_mask=attention_mask,
        )
        logits = add_final_logits_bias(self.bart, self.bart.lm_head(dec.last_hidden_state))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return {
            "loss": loss,
            "logits": logits,
            "c_entity": c_entity,
            "encoder_hidden_states": enc.last_hidden_state,
            "h_kg_nodes": h_kg,
        }


class FullDualOutputGNNModel(nn.Module):
    VARIANT = "full_dual_output"

    def __init__(self, bart_model, num_relations, gnn_hidden=384, num_bases=30):
        super().__init__()
        hidden_dim = bart_model.config.d_model
        self.bart = bart_model
        self.rgcn = DualOutputRGCN(hidden_dim, gnn_hidden, hidden_dim, num_relations, num_bases)
        self.kg_cross_attention = KGCrossAttention(hidden_dim)

    def forward(self, input_ids, attention_mask, decoder_input_ids,
                graph_x, graph_edge_index, graph_edge_type, graph_batch, labels=None):
        enc = self.bart.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_kg, c_entity = self.rgcn(graph_x, graph_edge_index, graph_edge_type)
        h_kg_pad, kg_mask = pad_kg_nodes(h_kg, graph_batch, input_ids.size(0))
        dec = self.bart.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=enc.last_hidden_state,
            encoder_attention_mask=attention_mask,
        )
        h_fused, gate_values, _ = self.kg_cross_attention(dec.last_hidden_state, h_kg_pad, kg_mask)
        logits = add_final_logits_bias(self.bart, self.bart.lm_head(h_fused))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return {
            "loss": loss,
            "logits": logits,
            "gate_values": gate_values,
            "c_entity": c_entity,
            "encoder_hidden_states": enc.last_hidden_state,
            "h_kg_nodes": h_kg,
        }


# ============================================================
# Tokenization / spans / loader
# ============================================================

def shift_tokens_right(labels, pad_id, dec_start_id):
    shifted = labels.clone()
    shifted[shifted == -100] = pad_id
    return torch.cat([
        torch.full((shifted.size(0), 1), dec_start_id, dtype=torch.long, device=shifted.device),
        shifted[:, :-1],
    ], dim=1)


def find_entity_token_spans(linearized_text, entity_names, tokenizer):
    full_ids = tokenizer.encode(linearized_text, add_special_tokens=True)
    spans = []
    used = set()

    for name in entity_names:
        raw = str(name).strip()
        clean = raw.replace("_", " ").strip()
        surfaces = []
        for s in [raw, clean, " " + raw, " " + clean]:
            if s and s not in surfaces:
                surfaces.append(s)
        candidates = [tokenizer.encode(s, add_special_tokens=False) for s in surfaces]
        candidates = sorted([c for c in candidates if c], key=len, reverse=True)

        found = False
        for ent_ids in candidates:
            m = len(ent_ids)
            for start in range(0, len(full_ids) - m + 1):
                if any(k in used for k in range(start, start + m)):
                    continue
                if full_ids[start:start + m] == ent_ids:
                    spans.append((start, start + m))
                    for k in range(start, start + m):
                        used.add(k)
                    found = True
                    break
            if found:
                break
        if not found:
            spans.append((-1, -1))
    return spans


def compute_alignment_loss(enc_hidden, h_kg, batch_vec, ent_starts, ent_ends):
    seq_len = enc_hidden.size(1)
    valid_mask = (ent_starts >= 0) & (ent_ends > ent_starts) & (ent_ends <= seq_len)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=enc_hidden.device)

    bi = batch_vec[valid_mask]
    h_gnn_valid = h_kg[valid_mask]
    starts = ent_starts[valid_mask]
    ends = ent_ends[valid_mask]

    bart_ents = []
    for j in range(h_gnn_valid.size(0)):
        s, e = int(starts[j].item()), int(ends[j].item())
        bart_ents.append(enc_hidden[bi[j], s:e].mean(dim=0))
    bart_ents = torch.stack(bart_ents)

    cos = nn.CosineSimilarity(dim=-1)
    return (1.0 - cos(h_gnn_valid, bart_ents.detach())).mean()


def attach_entity_labels(data, graphs):
    for split_name in ["train", "dev", "test"]:
        for i, ex in enumerate(data[split_name]):
            g = graphs[split_name][i]
            if hasattr(g, "entity_names") and g.entity_names:
                ref_lower = str(ex.get("target", "")).lower()
                labels = []
                for name in g.entity_names:
                    raw = str(name).lower().strip()
                    clean = raw.replace("_", " ").strip()
                    labels.append(1.0 if (raw in ref_lower or clean in ref_lower) else 0.0)
                g.entity_labels = torch.tensor(labels, dtype=torch.float)
            else:
                g.entity_labels = torch.zeros(g.x.size(0), dtype=torch.float)


class CombinedLoader:
    def __init__(self, text_examples, graphs, tokenizer, batch_size,
                 shuffle=True, max_input_len=256, max_target_len=128,
                 compute_spans=False):
        assert len(text_examples) == len(graphs), f"Data/graph length mismatch: {len(text_examples)} vs {len(graphs)}"
        self.graphs = graphs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(text_examples)))

        ids, masks, labs = [], [], []
        for i, ex in enumerate(text_examples):
            enc = tokenizer(
                ex["linearized"], max_length=max_input_len,
                truncation=True, padding="max_length", return_tensors="pt"
            )
            dec = tokenizer(
                text_target=ex["target"], max_length=max_target_len,
                truncation=True, padding="max_length", return_tensors="pt"
            )
            label = dec["input_ids"].squeeze(0)
            label[label == tokenizer.pad_token_id] = -100
            ids.append(enc["input_ids"].squeeze(0))
            masks.append(enc["attention_mask"].squeeze(0))
            labs.append(label)

            if compute_spans:
                g = graphs[i]
                if hasattr(g, "entity_names") and g.entity_names:
                    spans = find_entity_token_spans(ex["linearized"], g.entity_names, tokenizer)
                    g.ent_tok_start = torch.tensor([s for s, e in spans], dtype=torch.long)
                    g.ent_tok_end = torch.tensor([e for s, e in spans], dtype=torch.long)
                else:
                    n = g.x.size(0)
                    g.ent_tok_start = torch.full((n,), -1, dtype=torch.long)
                    g.ent_tok_end = torch.full((n,), -1, dtype=torch.long)

        self.input_ids = torch.stack(ids)
        self.attention_masks = torch.stack(masks)
        self.labels = torch.stack(labs)
        print(f"Pre-tokenized {len(text_examples)} examples" + (" + spans" if compute_spans else ""))

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = self.indices[:]
        if self.shuffle:
            random.shuffle(order)
        for start in range(0, len(order), self.batch_size):
            bi = order[start:start + self.batch_size]
            text = {
                "input_ids": self.input_ids[bi],
                "attention_mask": self.attention_masks[bi],
                "labels": self.labels[bi],
            }
            graph = PyGBatch.from_data_list([self.graphs[i] for i in bi])
            yield text, graph


# ============================================================
# Loss / validation
# ============================================================

def compute_total_loss(out, gb, variant, beta=0.5, gamma=0.3):
    if out["loss"] is None:
        raise ValueError("out['loss'] is None. Labels must be provided during training.")
    device = out["loss"].device
    L_task = out["loss"]
    L_con = torch.tensor(0.0, device=device)
    L_align = torch.tensor(0.0, device=device)

    if variant in ["constraint_only", "full_dual_output"] and hasattr(gb, "entity_labels"):
        labels = gb.entity_labels.float().to(device)
        c_ent = out["c_entity"].squeeze(-1)
        if labels.numel() == c_ent.numel():
            L_con = F.binary_cross_entropy_with_logits(c_ent, labels)

    if variant in ["fusion_only", "full_dual_output"] and hasattr(gb, "ent_tok_start") and hasattr(gb, "ent_tok_end"):
        L_align = compute_alignment_loss(
            out["encoder_hidden_states"],
            out["h_kg_nodes"],
            gb.batch,
            gb.ent_tok_start.to(device),
            gb.ent_tok_end.to(device),
        )

    total = L_task + beta * L_con + gamma * L_align
    logs = {
        "L_total": float(total.detach().item()),
        "L_task": float(L_task.detach().item()),
        "L_constraint": float(L_con.detach().item()),
        "L_align": float(L_align.detach().item()),
    }
    return total, logs


def evaluate_loss(model, loader, tokenizer, bart_model, device, variant, beta, gamma):
    model.eval()
    pad_id = tokenizer.pad_token_id
    dec_start_id = bart_model.config.decoder_start_token_id
    sums = {"L_total": 0.0, "L_task": 0.0, "L_constraint": 0.0, "L_align": 0.0}

    with torch.no_grad():
        for tb, gb in loader:
            tb = {k: v.to(device) for k, v in tb.items()}
            gb = gb.to(device)
            dec_ids = shift_tokens_right(tb["labels"], pad_id, dec_start_id)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(
                    input_ids=tb["input_ids"], attention_mask=tb["attention_mask"],
                    decoder_input_ids=dec_ids,
                    graph_x=gb.x, graph_edge_index=gb.edge_index,
                    graph_edge_type=gb.edge_type, graph_batch=gb.batch,
                    labels=tb["labels"],
                )
                _, logs = compute_total_loss(out, gb, variant, beta=beta, gamma=gamma)
            for k in sums:
                sums[k] += logs[k]
    return {k: v / max(len(loader), 1) for k, v in sums.items()}


# ============================================================
# True-resume training
# ============================================================

def _move_optimizer_state_to_device(optimizer, device: str):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_training_checkpoint(path, model, optimizer, scaler, epoch, best_val, history, variant, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "variant": variant,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_total": float(best_val),
        "history": history,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_training_checkpoint(path, model, optimizer=None, scaler=None, device="cuda", strict=False):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            _move_optimizer_state_to_device(optimizer, device)
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        finished_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val_total", float("inf")))
        history = ckpt.get("history", [])
        print(f"Loaded resume checkpoint: {path}")
        print(f"Checkpoint epoch: {finished_epoch}")
        print(f"Best val total so far: {best_val:.4f}")
        return finished_epoch + 1, best_val, history

    model.load_state_dict(ckpt, strict=strict)
    print(f"Loaded old weight-only checkpoint: {path}")
    return 1, float("inf"), []


def run_train_resume(model, data, graphs, tokenizer, bart_model, checkpoint_dir, variant,
                     device, num_epochs=3, batch_size=8, lr=1e-4,
                     bart_lr=3e-5, kg_lr=1e-4, grad_accum=1,
                     beta=0.5, gamma=0.3, max_grad_norm=1.0,
                     resume_from_checkpoint=None, load_best_at_end=False):
    variant_ckpt = os.path.join(checkpoint_dir, variant)
    os.makedirs(variant_ckpt, exist_ok=True)
    best_path = os.path.join(variant_ckpt, "model_best.pt")
    final_path = os.path.join(variant_ckpt, "model_final.pt")
    history_json = os.path.join(variant_ckpt, "history.json")
    history_csv = os.path.join(variant_ckpt, "history.csv")

    compute_spans = variant in ["fusion_only", "full_dual_output"]
    train_loader = CombinedLoader(data["train"], graphs["train"], tokenizer,
                                  batch_size, shuffle=True, compute_spans=compute_spans)
    dev_loader = CombinedLoader(data["dev"], graphs["dev"], tokenizer,
                                batch_size * 2, shuffle=False, compute_spans=compute_spans)

    optimizer = build_trainable_optimizer(model, bart_lr=bart_lr, kg_lr=kg_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    start_epoch = 1
    best_val = float("inf")
    history = []
    if resume_from_checkpoint is not None:
        start_epoch, best_val, history = load_training_checkpoint(
            resume_from_checkpoint, model, optimizer=optimizer, scaler=scaler, device=device, strict=False
        )

    if start_epoch > num_epochs:
        print(f"Checkpoint already finished epoch {start_epoch - 1}; target num_epochs={num_epochs}.")
        print("No additional training will run. Increase EPOCHS if you want to continue.")
        return model, history, {"best": best_path, "final": final_path,
                               "history_json": history_json, "history_csv": history_csv}

    pad_id = tokenizer.pad_token_id
    dec_start_id = bart_model.config.decoder_start_token_id
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    print(f"Training {variant}: start_epoch={start_epoch}, target_epochs={num_epochs}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,} / total {count_total_parameters(model):,}")
    print(f"BART LR: {bart_lr} | KG LR: {kg_lr} | batch_size={batch_size} | grad_accum={grad_accum}")

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")
        model.train()
        running = {"L_total": 0.0, "L_task": 0.0, "L_constraint": 0.0, "L_align": 0.0}
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()

        for step, (tb, gb) in enumerate(train_loader):
            tb = {k: v.to(device) for k, v in tb.items()}
            gb = gb.to(device)
            dec_ids = shift_tokens_right(tb["labels"], pad_id, dec_start_id)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(
                    input_ids=tb["input_ids"], attention_mask=tb["attention_mask"],
                    decoder_input_ids=dec_ids,
                    graph_x=gb.x, graph_edge_index=gb.edge_index,
                    graph_edge_type=gb.edge_type, graph_batch=gb.batch,
                    labels=tb["labels"],
                )
                loss, logs = compute_total_loss(out, gb, variant, beta=beta, gamma=gamma)
                loss_to_backprop = loss / grad_accum

            scaler.scale(loss_to_backprop).backward()
            is_update_step = ((step + 1) % grad_accum == 0) or ((step + 1) == len(train_loader))
            if is_update_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            for k in running:
                running[k] += logs[k]
            if (step + 1) % 300 == 0:
                msg = " | ".join(f"{k}={running[k]/(step+1):.4f}" for k in running)
                print(f"epoch {epoch} step {step+1}/{len(train_loader)}: {msg}")

        train_logs = {k: v / max(len(train_loader), 1) for k, v in running.items()}
        val_logs = evaluate_loss(model, dev_loader, tokenizer, bart_model, device, variant, beta, gamma)
        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "train_L_total": train_logs["L_total"],
            "train_L_task": train_logs["L_task"],
            "train_L_constraint": train_logs["L_constraint"],
            "train_L_align": train_logs["L_align"],
            "val_L_total": val_logs["L_total"],
            "val_L_task": val_logs["L_task"],
            "val_L_constraint": val_logs["L_constraint"],
            "val_L_align": val_logs["L_align"],
            "elapsed_sec": elapsed,
        }
        history.append(row)
        print(
            f"Epoch {epoch}: train_total={row['train_L_total']:.4f} "
            f"train_task={row['train_L_task']:.4f} train_con={row['train_L_constraint']:.4f} "
            f"train_align={row['train_L_align']:.4f} | val_total={row['val_L_total']:.4f} "
            f"val_task={row['val_L_task']:.4f} val_con={row['val_L_constraint']:.4f} "
            f"val_align={row['val_L_align']:.4f} | time={elapsed/60:.1f} min"
        )

        if val_logs["L_total"] < best_val:
            best_val = val_logs["L_total"]
            save_training_checkpoint(best_path, model, optimizer, scaler, epoch, best_val, history, variant)
            print(f"  saved best -> {best_path}")

        save_training_checkpoint(final_path, model, optimizer, scaler, epoch, best_val, history, variant)
        print(f"  updated final/resume checkpoint -> {final_path}")

        with open(history_json, "w") as f:
            json.dump(history, f, indent=2)
        try:
            import pandas as pd
            pd.DataFrame(history).to_csv(history_csv, index=False)
        except Exception as e:
            print("Could not save history CSV:", e)

    print(f"Training finished. Best val total loss: {best_val:.4f}")
    print(f"Best checkpoint:  {best_path}")
    print(f"Final checkpoint: {final_path}")

    if load_best_at_end:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print("Loaded best checkpoint because load_best_at_end=True.")

    return model, history, {"best": best_path, "final": final_path,
                           "history_json": history_json, "history_csv": history_csv}


def load_variant_checkpoint_resume(model, checkpoint_path, device="cuda", strict=False):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}: {checkpoint_path}")
    else:
        state = ckpt
        print(f"Loaded old weight-only checkpoint: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(state, strict=strict)
    model.to(device)
    model.eval()
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
    return model


# ============================================================
# Generation / metrics / per-sample evaluation
# ============================================================

@torch.no_grad()
def generate_one(model, tokenizer, ex, graph, device, variant, max_gen_len=128, alpha=1.0, max_input_len=256):
    model.eval()
    enc_inputs = tokenizer(
        ex["linearized"], max_length=max_input_len, truncation=True,
        padding="max_length", return_tensors="pt"
    )
    inp_ids = enc_inputs["input_ids"].to(device)
    att_mask = enc_inputs["attention_mask"].to(device)

    gx = graph.x.to(device)
    gei = graph.edge_index.to(device)
    get = graph.edge_type.to(device)
    gbatch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
    enames = graph.entity_names if hasattr(graph, "entity_names") else []

    enc_out = model.bart.model.encoder(input_ids=inp_ids, attention_mask=att_mask)
    h_kg_all, c_ent = model.rgcn(gx, gei, get)

    trie = None
    if variant in ["constraint_only", "full_dual_output"]:
        scores = signed_entity_scores(c_ent).detach().cpu()
        trie = TrieMap(enames, scores, tokenizer)

    h_kg_pad, kg_mask = None, None
    if variant in ["fusion_only", "full_dual_output"]:
        h_kg_pad, kg_mask = pad_kg_nodes(h_kg_all, gbatch, 1)

    dec_start_id = model.bart.config.decoder_start_token_id
    eos_id = model.bart.config.eos_token_id
    gen_ids = [dec_start_id]
    past = None

    for _ in range(max_gen_len - 1):
        dec_in = torch.tensor([[gen_ids[-1]]], dtype=torch.long, device=device)
        dec_out = model.bart.model.decoder(
            input_ids=dec_in,
            encoder_hidden_states=enc_out.last_hidden_state,
            encoder_attention_mask=att_mask,
            past_key_values=past,
            use_cache=True,
        )
        past = dec_out.past_key_values
        h_dec = dec_out.last_hidden_state
        if variant in ["fusion_only", "full_dual_output"]:
            h_dec, _, _ = model.kg_cross_attention(h_dec, h_kg_pad, kg_mask)
        logits = add_final_logits_bias(model.bart, model.bart.lm_head(h_dec))[:, -1, :]
        if trie is not None:
            c_tok = trie.get_constraint_vector(gen_ids).to(device)
            logits = logits + alpha * c_tok.unsqueeze(0)
        next_id = int(logits.argmax(dim=-1).item())
        if next_id == eos_id:
            break
        gen_ids.append(next_id)

    return tokenizer.decode(gen_ids[1:], skip_special_tokens=True)


def _normalize_entity_surface(e):
    e = str(e).strip()
    return [e, e.replace("_", " ")]


def _input_entities_from_triples(triples):
    ents = set()
    for t in triples:
        if isinstance(t, dict):
            vals = [t.get("subject", ""), t.get("object", "")]
        else:
            vals = [t[0], t[2]]
        for v in vals:
            for s in _normalize_entity_surface(v):
                s = s.lower().strip()
                if s:
                    ents.add(s)
    return ents


def _found_entities_in_text(text, candidate_entities):
    text_lower = str(text).lower()
    found = set()
    for ent in candidate_entities:
        ent = str(ent).lower().strip()
        if not ent:
            continue
        raw = ent
        clean = ent.replace("_", " ")
        if raw in text_lower:
            found.add(raw)
        if clean in text_lower:
            found.add(clean)
    return found


def compute_per_sample_metrics(prediction, references, triples, global_entities=None):
    pred_tok = prediction.split()
    ref_tok = [r.split() for r in references]
    try:
        sample_bleu = corpus_bleu([ref_tok], [pred_tok], smoothing_function=SmoothingFunction().method1) * 100
    except Exception:
        sample_bleu = 0.0
    try:
        sample_meteor = float(nltk_meteor(ref_tok, pred_tok))
    except Exception:
        sample_meteor = 0.0
    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        sample_rouge = float(max(scorer.score(r, prediction)["rougeL"].fmeasure for r in references))
    except Exception:
        sample_rouge = 0.0

    input_ents = _input_entities_from_triples(triples)
    if global_entities:
        candidate_ents = set()
        for e in global_entities:
            for s in _normalize_entity_surface(e):
                s = s.lower().strip()
                if s:
                    candidate_ents.add(s)
        output_ents = _found_entities_in_text(prediction, candidate_ents)
    else:
        output_ents = set()

    pred_lower = str(prediction).lower()
    found_input = {e for e in input_ents if e in pred_lower}
    missing = input_ents - found_input
    if global_entities:
        hallucinated = output_ents - input_ents
        entity_precision = len(found_input) / len(output_ents) if output_ents else 1.0
    else:
        hallucinated = set()
        entity_precision = 1.0
    entity_recall = len(found_input) / len(input_ents) if input_ents else 1.0
    hallucination_rate = 1.0 - entity_precision

    if hallucinated:
        weakness_type = "entity_hallucination"
    elif entity_recall < 0.5:
        weakness_type = "entity_omission"
    elif sample_bleu < 30:
        weakness_type = "low_text_overlap"
    else:
        weakness_type = "ok_or_minor"

    return {
        "bleu": sample_bleu,
        "meteor": sample_meteor,
        "rouge_l": sample_rouge,
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "hallucination_rate": hallucination_rate,
        "input_entities": sorted(input_ents),
        "output_entities": sorted(output_ents),
        "missing_entities": sorted(missing),
        "hallucinated_entities": sorted(hallucinated),
        "weakness_type": weakness_type,
    }


def compute_generation_metrics(predictions, references_list, triples_list=None, global_entities=None):
    import nltk
    for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

    pred_tok = [p.split() for p in predictions]
    ref_tok = [[r.split() for r in refs] for refs in references_list]
    results = {}
    results["bleu"] = corpus_bleu(ref_tok, pred_tok, smoothing_function=SmoothingFunction().method1) * 100
    results["meteor"] = float(np.mean([nltk_meteor([r.split() for r in refs], pred.split()) for pred, refs in zip(predictions, references_list)]))
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results["rouge_l"] = float(np.mean([max(scorer.score(r, pred)["rougeL"].fmeasure for r in refs) for pred, refs in zip(predictions, references_list)]))

    if triples_list is not None:
        per_sample = [
            compute_per_sample_metrics(pred, refs, triples, global_entities)
            for pred, refs, triples in zip(predictions, references_list, triples_list)
        ]
        results["entity_precision"] = float(np.mean([x["entity_precision"] for x in per_sample]))
        results["entity_recall"] = float(np.mean([x["entity_recall"] for x in per_sample]))
        results["hallucination_rate"] = float(np.mean([x["hallucination_rate"] for x in per_sample]))
    return results


def print_metrics(m):
    print(f"BLEU-4:      {m.get('bleu', 0):.2f}")
    print(f"METEOR:      {m.get('meteor', 0):.4f}")
    print(f"ROUGE-L:     {m.get('rouge_l', 0):.4f}")
    if "entity_precision" in m:
        print(f"Ent Prec:    {m['entity_precision']:.4f}")
        print(f"Ent Recall:  {m['entity_recall']:.4f}")
        print(f"Halluc Rate: {m['hallucination_rate']:.4f}")


def run_generation_eval(model, data, graphs, tokenizer, processed_dir, device, variant,
                        global_entity_set=None, split="dev", max_gen_len=128, alpha=1.0,
                        output_tag=None):
    predictions = []
    print(f"Generating {variant} on {split} set: {len(data[split])} examples")
    t0 = time.time()
    for i, ex in enumerate(data[split]):
        pred = generate_one(model, tokenizer, ex, graphs[split][i], device, variant,
                            max_gen_len=max_gen_len, alpha=alpha)
        predictions.append(pred)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(data[split])} ({time.time()-t0:.0f}s)")

    refs = [ex["all_targets"] for ex in data[split]]
    triples = [ex["triples"] for ex in data[split]]
    metrics = compute_generation_metrics(predictions, refs, triples, global_entity_set)

    per_sample_rows = []
    for i, (pred, ex) in enumerate(zip(predictions, data[split])):
        ps = compute_per_sample_metrics(pred, ex["all_targets"], ex["triples"], global_entity_set)
        row = {
            "sample_id": i,
            "input_linearized": ex.get("linearized", ""),
            "input_triples": json.dumps(ex.get("triples", []), ensure_ascii=False),
            "reference": ex.get("target", ""),
            "all_references": json.dumps(ex.get("all_targets", []), ensure_ascii=False),
            "prediction": pred,
            **ps,
        }
        for key in ["input_entities", "output_entities", "missing_entities", "hallucinated_entities"]:
            row[key] = json.dumps(row[key], ensure_ascii=False)
        per_sample_rows.append(row)

    print_metrics(metrics)
    os.makedirs(processed_dir, exist_ok=True)
    tag = output_tag if output_tag else variant
    metrics_path = os.path.join(processed_dir, f"metrics_{tag}_{split}.json")
    pred_path = os.path.join(processed_dir, f"predictions_{tag}_{split}.json")
    per_json_path = os.path.join(processed_dir, f"per_sample_{tag}_{split}.json")
    per_csv_path = os.path.join(processed_dir, f"per_sample_{tag}_{split}.csv")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    with open(per_json_path, "w") as f:
        json.dump(per_sample_rows, f, indent=2, ensure_ascii=False)
    try:
        import pandas as pd
        pd.DataFrame(per_sample_rows).to_csv(per_csv_path, index=False)
    except Exception as e:
        print("Could not save per-sample CSV:", e)

    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved predictions -> {pred_path}")
    print(f"Saved per-sample JSON -> {per_json_path}")
    print(f"Saved per-sample CSV -> {per_csv_path}")

    print("\nSample predictions:")
    for j in range(min(5, len(predictions))):
        print(f"[{j}] PRED: {predictions[j]}")
        print(f"    REF:  {data[split][j]['target']}")

    return metrics, predictions, per_sample_rows


# ============================================================
# Negative training: distractor injection (graph-only)
# ============================================================

def inject_distractor_nodes(train_graphs, num_rel, min_d=1, max_d=3, seed=42, verbose=True):
    """Graph-only negative training for the constraint head.

    For each TRAINING graph, sample `min_d`..`max_d` distractor *triples* from
    other training graphs and append their entities as new nodes plus the
    bidirectional edge, exactly matching how `_triples_to_graph` builds graphs
    (forward edge type `r`, reverse edge type `r + num_rel`).

    Distractor node feature vectors are reused from the source graph's `x`, so no
    BART/tokenizer pass is needed here. Distractor entity names are NOT added to
    the linearized BART input, so:
      * `attach_entity_labels` later assigns label 0 to them (not in the target),
        giving the constraint head a genuine negative signal, and
      * `find_entity_token_spans` returns (-1, -1) for them, so the alignment
        loss automatically ignores distractor nodes.

    The function mutates `train_graphs` in place and returns it. Call it AFTER
    loading/slicing graphs and BEFORE `attach_entity_labels` / building loaders,
    on the TRAIN split only (never dev/test). Re-running the cell that calls
    `load_artifacts` reloads clean graphs from disk, so injection is not
    cumulative across re-runs.
    """
    rng = random.Random(seed)
    n = len(train_graphs)
    if n == 0:
        return train_graphs

    total_triples = 0
    total_nodes = 0
    touched = 0

    for i, g in enumerate(train_graphs):
        names = list(getattr(g, "entity_names", []) or [])
        existing = {str(e).lower().strip() for e in names}
        k = rng.randint(min_d, max_d)

        new_names, new_feats = [], []
        new_src, new_dst, new_etype = [], [], []
        name_to_new_idx = {}
        base = g.x.size(0)

        added_triples = 0
        attempts = 0
        max_attempts = max(k * 20, 20)

        while added_triples < k and attempts < max_attempts:
            attempts += 1
            j = rng.randrange(n)
            if j == i:
                continue
            dg = train_graphs[j]
            dnames = getattr(dg, "entity_names", None)
            if not dnames or dg.edge_index.numel() == 0:
                continue

            # Only forward edges are real triples (reverse edges use type >= num_rel).
            fwd = (dg.edge_type < num_rel).nonzero(as_tuple=True)[0]
            if fwd.numel() == 0:
                continue
            col = int(fwd[rng.randrange(int(fwd.numel()))].item())
            a = int(dg.edge_index[0, col].item())
            b = int(dg.edge_index[1, col].item())
            r = int(dg.edge_type[col].item())
            if a >= len(dnames) or b >= len(dnames):
                continue

            na, nb = str(dnames[a]), str(dnames[b])
            la, lb = na.lower().strip(), nb.lower().strip()
            # Require both endpoints to be genuinely new (not already real entities here).
            if not la or not lb or la in existing or lb in existing:
                continue

            for nm, lk, feat in [(na, la, dg.x[a]), (nb, lb, dg.x[b])]:
                if lk not in name_to_new_idx:
                    name_to_new_idx[lk] = base + len(new_names)
                    new_names.append(nm)
                    new_feats.append(feat.detach().clone())
                    existing.add(lk)
            ia, ib = name_to_new_idx[la], name_to_new_idx[lb]
            new_src += [ia, ib]
            new_dst += [ib, ia]
            new_etype += [r, r + num_rel]
            added_triples += 1

        if new_feats:
            add_x = torch.stack(new_feats, dim=0).to(dtype=g.x.dtype)
            g.x = torch.cat([g.x, add_x], dim=0)

            add_ei = torch.tensor([new_src, new_dst], dtype=torch.long)
            add_et = torch.tensor(new_etype, dtype=torch.long)
            if g.edge_index.numel():
                g.edge_index = torch.cat([g.edge_index, add_ei], dim=1)
                g.edge_type = torch.cat([g.edge_type, add_et], dim=0)
            else:
                g.edge_index = add_ei
                g.edge_type = add_et

            g.entity_names = names + new_names
            g.num_nodes = g.x.size(0)

            touched += 1
            total_triples += added_triples
            total_nodes += len(new_names)

    if verbose:
        print(
            f"Distractor injection: added {total_triples} triples / {total_nodes} nodes "
            f"across {touched}/{n} training graphs (min_d={min_d}, max_d={max_d}, seed={seed})."
        )
    return train_graphs
