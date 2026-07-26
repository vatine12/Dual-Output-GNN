# Architecture

## Notation

| Symbol | Meaning |
|---|---|
| `G = (V, E, R)` | input knowledge graph: entities, edges, relation types |
| `g_i^F` | shared R-GCN representation of entity `i` (fusion representation) |
| `Δg_i^C` | constraint-head correction for entity `i` |
| `g_i^C` | constraint representation, `g_i^F + γ·Δg_i^C` |
| `h_t^BART` | BART decoder state at step `t` |
| `h_t^F` | graph-fused decoder state at step `t` |
| `c_{t,i}` | coverage: 1 if entity `i` has already been realized |
| `z_t` | vocabulary logits, `LMHead(h_t^F)` |
| `τ`, `M` | selector confidence threshold, LM compatibility margin |

## 1. Shared graph trunk

Triples are converted to a multi-relational graph: each distinct entity becomes a node, each triple `(h, r, t)` a directed edge of type `r` plus its inverse `r + |R|`. Node features initialize from mean-pooled BART embeddings of the entity name, so nodes start in the language model's embedding space. Two R-GCN layers with basis decomposition produce `g_i^F`.

## 2. Fusion branch

The decoder attends to graph nodes through a separate cross-attention (BART's own encoder cross-attention is untouched), followed by a learned gate:

```
H_KG(t)   = softmax(Q(t) K^T / √d) V          Q from h_t^BART, K/V from {g_i^F}
λ(t)      = σ(W_g [h_t^BART ; H_KG(t)] + b_g)
h_t^F     = λ(t)·h_t^BART + (1 − λ(t))·H_KG(t)
```

Keeping the `λ·h_t^BART` term holds the fused state inside the language model's hidden space, so the pretrained output projection still reads it correctly.

## 3. Constraint head (the second head)

```
Δg_i^C = RGCN_C({g_j^F}, E)_i
g_i^C  = g_i^F + γ · Δg_i^C
```

A shallow relation-aware refinement specialised for the selection decision, with `γ` learned. The output projection is zero-initialised, so before training `g^C ≡ g^F` and the dual-head model reproduces the shared-representation model exactly — every subsequent change is attributable to the head.

**Why residual rather than replacement:** the fusion representation is already well trained; the constraint branch should adjust it, not relearn it. Verified empirically — at the selected checkpoint the learned correction has mean norm ≈ 0.72 with `γ` ≈ 0.89, i.e. an active but bounded refinement.

## 4. Adaptive entity selector

At each decoding step the selector scores `{NONE} ∪ {e_1..e_N}`:

```
s_{t,i}    = MLP_entity([h_t^F ; g_i^C ; c_{t,i}])      1537 → 256 → 1
s_{t,NONE} = MLP_NONE(h_t^F)                             768 → 256 → 1
```

- **`NONE`** is essential: most positions are ordinary language positions. Without an abstention action the controller would insert entities far too often.
- **Coverage** `c_{t,i}` marks already-realized entities, which are masked from selection before the argmax — this prevents repetition and stops a high-scoring covered entity from crowding out an uncovered one.
- The selector reads `h_t^F` (the fused state, which already carries graph context), while the compatibility gate reads `z_t = LMHead(h_t^F)`. Two different views of the same state, used for two different decisions.

### Training

Backbone frozen: BART, shared R-GCN, fusion, and the `NONE` scorer. Trainable: constraint head, `γ`, entity MLP.

Labels come from gold entity spans in the reference text, aligned to decoder positions:

```
y_t = 0      NONE
    = i + 1  entity i starts at t
    = −100   inside an entity span (ignored — exact commitment handles continuation)
```

Weighted cross-entropy with `w_NONE = 0.3` (most positions are `NONE`). The selector trains as a pure classifier on teacher-forced states rather than back-propagating through constrained decoding: simpler, and it avoids creating the exposure-bias interaction that training-through-decoding would introduce. The confidence and compatibility gates manage the residual train/inference state mismatch.

### Checkpoint selection

Select on **entity-start recall**, not intrinsic F1. The two error types have asymmetric downstream costs:

- a *false* activation is often rejected downstream by the confidence gate, the compatibility gate, or coverage masking;
- a *missed* activation is an unrecoverable omission — nothing later in the pipeline can insert it.

Choosing by F1 favours precision-heavy checkpoints that fire too rarely. This is visible in the data: a higher-F1 checkpoint produced measurably worse generation than a lower-F1, higher-recall one.

## 5. Decoding

```
for each step t:
    if committed to entity i:
        emit next token of entity i's exact sequence
    else:
        z_t = LMHead(h_t^F)
        mask covered / ineligible entities
        p   = softmax([s_NONE, s_entities])
        if p(best entity) ≥ τ and max(z_t) − z_t(first token) ≤ M:
            emit first token, commit to the rest of the sequence
        else:
            z_t ← hard_completion_mask(z_t)     # Hard-v2 backstop
            emit argmax(z_t)
```

**Exact commitment** matters because entities frequently share prefixes: having selected `Pontiac_Rageous`, the model must not complete it as a different `Pontiac ...` entity. Committing the whole token sequence removes that failure mode entirely.

**Hard-v2** is independent of the selector and handles the complementary case — the language model starts a graph-entity name on its own initiative. A trie built from cleaned entity names restricts continuations once ≥ 2 sub-tokens match (or the prefix is unique), releases at a completed name, and escapes if the unconstrained model prefers an alternative by more than 10 logits.

Two properties of the trie matter:

1. **Both tokenizations.** Byte-level BPE assigns different ids to a word at sentence start vs. mid-sentence (`Pont` vs `ĠPont`). Both variants are inserted; indexing only the first makes the mechanism inert mid-sentence.
2. **Names only.** Dates, measurements and numeric literals are excluded, and parenthetical disambiguators are stripped (`Harry_Carey_(actor_born_1878)` → `Harry Carey`). Otherwise completion forces raw knowledge-base labels into the text — `"13 October 1964-10-13"` — which inflates entity metrics while degrading the actual output.

## 6. Frozen operating point

| Parameter | Value | Selected on |
|---|---|---|
| `τ` selector threshold | 0.70 | development, trigger-budget matched |
| `M` compatibility margin | 8.0 | development |
| Hard-v2 minimum depth | 2 sub-tokens | development |
| Hard-v2 escape margin | 10.0 logits | development |

When comparing two selector variants, match the **trigger budget**, not just the threshold. Different representations produce differently calibrated scores, so a fixed τ places them at different points on their precision/recall curves — an apparent quality difference that is really an operating-point difference.
