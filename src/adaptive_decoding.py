"""
adaptive_decoding.py — the decoding-side components of the Dual-Head Adaptive GNN.

Three pieces, in the order they act during generation:

1. `EntitySelector`      — at every decoder step, scores {NONE} u {graph entities} from
                           [fused decoder state ; entity representation ; coverage] and
                           decides whether an entity mention should START here.
2. `commit_entity`       — once an entity is selected, its exact token sequence is forced,
                           so the realization cannot drift to a different entity sharing a prefix.
3. `TrieMapV2` +
   `hard_completion_mask`— backstop: when the language model begins a graph-entity prefix on
                           its own, restrict continuations to valid entity names.

Names only: dates, numbers and other literals are excluded from the completion trie, and
DBpedia-style disambiguation parentheticals are stripped, so the mechanism never forces raw
knowledge-base labels into the surface text.
"""

import re
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------------
# 1. Decoder-conditioned entity selector
# ----------------------------------------------------------------------------------

class EntitySelector(nn.Module):
    """Scores {NONE} u {e_1..e_N} at a decoding step.

    h    : (L, d)    fused decoder states
    ents : (N, d)    entity representations (g^F shared, or g^C from the constraint head)
    cov  : (L, N)    1.0 where the entity has already been realized
    returns (L, 1 + N); column 0 is NONE.
    """

    def __init__(self, d: int = 768, hid: int = 256):
        super().__init__()
        self.ent_mlp = nn.Sequential(nn.Linear(2 * d + 1, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.none_mlp = nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, h, ents, cov):
        L, d = h.shape
        N = ents.shape[0]
        he = torch.cat([h.unsqueeze(1).expand(L, N, d),
                        ents.unsqueeze(0).expand(L, N, d),
                        cov.unsqueeze(-1)], dim=-1)
        return torch.cat([self.none_mlp(h), self.ent_mlp(he).squeeze(-1)], dim=-1)


class ConstraintHead(nn.Module):
    """Residual constraint-specific graph branch: g^C = g^F + gamma * RGCN_C(g^F).

    The output projection is zero-initialized, so an untrained head reproduces the shared
    representation exactly and training starts from a function-preserving point.
    """

    def __init__(self, d: int = 768, hidden: int = 256, num_relations: int = 824,
                 num_bases: int = 8, dropout: float = 0.1):
        super().__init__()
        from torch_geometric.nn import RGCNConv
        self.conv = RGCNConv(d, hidden, num_relations, num_bases=num_bases)
        self.out = nn.Linear(hidden, d)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, g_shared, edge_index, edge_type):
        delta = self.out(self.dropout(torch.relu(self.conv(g_shared, edge_index, edge_type))))
        return g_shared + self.gamma * delta


# ----------------------------------------------------------------------------------
# 2 + 3. Name-only completion trie and hard masking
# ----------------------------------------------------------------------------------

_LITERAL_PATTERNS = [r'^[\d\s.,:/\-+%°"]*$', r'^\d{3,4}-\d{1,2}-\d{1,2}', r'^\d+(\.\d+)?$']


def is_literal(name: str) -> bool:
    """True for dates, measurements and other non-name values, which never enter the trie."""
    n = str(name).strip().strip('"').strip("'").strip()
    return len(n) < 2 or any(re.match(p, n) for p in _LITERAL_PATTERNS)


def clean_surface(name: str) -> str:
    """`Harry_Carey_(actor_born_1878)` -> `Harry Carey`."""
    s = str(name).strip().strip('"').strip("'").replace('_', ' ')
    s = re.sub(r'\s*\([^)]*\)', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


class _Node:
    __slots__ = ('ch', 'terminal', 'nterm')

    def __init__(self):
        self.ch = {}
        self.terminal = False
        self.nterm = 0


class TrieMapV2:
    """Token trie over cleaned entity names, in both sentence-initial and mid-sentence
    (leading-space) tokenizations — byte-level BPE assigns different ids to each."""

    def __init__(self, entity_names, tokenizer):
        self.root = _Node()
        self.kept = {}
        for idx, name in enumerate(entity_names):
            if is_literal(name):
                continue
            base = clean_surface(name)
            if not base or is_literal(base):
                continue
            self.kept[idx] = base
            for variant in (base, ' ' + base):
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if not ids:
                    continue
                node = self.root
                for tid in ids:
                    node = node.ch.setdefault(int(tid), _Node())
                node.terminal = True
        self._cache(self.root)

    def _cache(self, node):
        node.nterm = (1 if node.terminal else 0) + sum(self._cache(c) for c in node.ch.values())
        return node.nterm

    def suffix_matches(self, generated_ids, lookback: int = 20):
        """All trie nodes reachable by reading a suffix of the generated tokens."""
        out = []
        for start in range(max(0, len(generated_ids) - lookback), len(generated_ids)):
            node, ok = self.root, True
            for pos in range(start, len(generated_ids)):
                tid = int(generated_ids[pos])
                if tid not in node.ch:
                    ok = False
                    break
                node = node.ch[tid]
            if ok and node is not self.root:
                out.append((len(generated_ids) - start, node))
        return out


def hard_completion_mask(logits, trie: TrieMapV2, generated_ids,
                         min_depth: int = 2, escape_margin: float = 10.0):
    """Restrict continuations when the model is mid-way through a graph-entity name.

    Fires only on an unambiguous prefix (>= `min_depth` matched sub-tokens, or a unique
    entity). Releases at a completed name (`terminal`), and escapes entirely when the
    unconstrained model prefers something else by more than `escape_margin` logits.
    """
    best = None
    for depth, node in trie.suffix_matches(generated_ids):
        if node.terminal or not node.ch:
            continue
        if (depth >= min_depth or node.nterm == 1) and (best is None or depth > best[0]):
            best = (depth, node)
    if best is None:
        return logits, False
    legal = list(best[1].ch.keys())
    if float(logits.max()) - max(float(logits[t]) for t in legal) > escape_margin:
        return logits, False
    mask = torch.full_like(logits, -1e9)
    mask[legal] = 0.0
    return logits + mask, True


def entity_token_ids(trie: TrieMapV2, tokenizer, entity_index: int, at_sequence_start: bool):
    """Exact token sequence to commit to for a selected entity."""
    base = trie.kept.get(entity_index)
    if not base:
        return None
    return tokenizer.encode(base if at_sequence_start else ' ' + base, add_special_tokens=False)


# ----------------------------------------------------------------------------------
# Frozen operating point (selected on development data)
# ----------------------------------------------------------------------------------

DEFAULT_CONFIG = {
    'selector_threshold': 0.70,   # tau: minimum selector probability to trigger a mention
    'compatibility_margin': 8.0,  # M: max logit gap between the LM's top token and the entity's first token
    'hard_v2_min_depth': 2,
    'hard_v2_escape_margin': 10.0,
}
