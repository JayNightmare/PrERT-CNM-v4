# Contextual Neural Memory (CNM)

Retrieval-augmented memory over the harmonised standards catalogue. Given a
policy clause, the CNM retrieves the top-`k` semantically-similar controls
from Phase 1's catalogue and concatenates their text into the PrivBERT input,
giving the classifier standards-aware context at inference time.

## Design

Formally, given clause $x_i$, tokeniser $T$, PrivBERT encoder $E_\theta$, and
memory index $\mathcal{M} = \{(c_j, e_j)\}_{j=1}^{|\mathcal{M}|}$ where $c_j$
is the control text and $e_j \in \mathbb{R}^{D}$ is its embedding under encoder
$\phi$:

1. Query embedding: $q_i = \phi(x_i)$
2. Top-k retrieval: $\mathcal{R}_k(x_i) = \operatorname*{argtop-k}_{j} \cos(q_i, e_j)$
3. Augmented input: $\tilde{x}_i = x_i \; [\text{SEP}] \; c_{r_1} \; [\text{SEP}] \; \ldots \; [\text{SEP}] \; c_{r_k}$
4. Classification: $p_i = \operatorname{softmax}(E_\theta(T(\tilde{x}_i)))$

`k = 0` disables retrieval and the wrapper is equivalent to the plain
`PrivacyBertClassifier` baseline.

## Files

| File | Purpose |
|------|---------|
| `embed.py` | Encoder facade (sentence-transformers preferred, transformers fallback) |
| `memory.py` | `MemoryEntry`, `MemoryIndex`, `build_memory_index`, `load_memory_index` |
| `retriever.py` | `Retriever` — cosine top-k over the numpy index with query cache |
| `classifier.py` | `CNMPrivacyBertClassifier` — subclass of `PrivacyBertClassifier` that augments inputs |
| `pipeline.py` | `run_phase3_cnm_pipeline`, `run_cnm_ablation` |
| `cli.py` | `prert-cnm-build`, `prert-cnm-phase3`, `prert-cnm-ablation` |
| `discover_controls.py` | Finds a plausible Phase 1 controls file to use as input |
| `tests/` | Deterministic unit tests |

## Running

### 1. Find your Phase 1 controls file

```bash
python -m prert.phase3.cnm.discover_controls
```

This walks `artifacts/` and prints candidate JSON/JSONL files, showing keys
and sample entries so you can pick the right one.

### 2. Build the memory index (one-time, ~2 min on CPU for 683 controls)

```bash
python -m prert.phase3.cnm.cli build \
    --controls-path artifacts/phase-1/controls.jsonl \
    --output-dir artifacts/cnm-index
```

The default encoder is `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
Override with `--encoder`. If sentence-transformers is not installed, a
mean-pooled transformers fallback is used automatically.

Outputs in `artifacts/cnm-index/`:
  - `embeddings.npz` — float32 matrix (N, 384), L2-normalised
  - `entries.jsonl` — MemoryEntry records
  - `index_metadata.json` — encoder name, count, corpus hash

### 3. Run Phase 3 with CNM enabled (single k)

```bash
python -m prert.phase3.cnm.cli phase3 \
    --cnm-index artifacts/cnm-index \
    --top-k 5 \
    --output-dir artifacts/phase-3-cnm-k5 \
    --run-id cnm-k5
```

Produces the same artefact tree as `prert-phase3-freeze` (predictions,
calibration, bootstrap, Bayesian scoring). Adds `cnm_config.json` inside
`classifier_checkpoint/privacybert/`.

### 4. Full top-k ablation (paper table)

```bash
python -m prert.phase3.cnm.cli ablation \
    --cnm-index artifacts/cnm-index \
    --k-values 0,1,3,5,10 \
    --output-dir artifacts/phase-3-cnm-ablation
```

Writes per-k subdirectories `k=0/`, `k=1/`, ... and a top-level
`ablation_summary.json` with the val/test macro-F1, accuracy, and Bayesian
primary score for each k. `k=0` is the plain-PrivBERT baseline.

### 5. Retrieval traces (optional)

Pass `--trace-output artifacts/cnm-traces.jsonl` to record, for every
training and inference clause, which controls were retrieved with what
similarity scores. Useful for the paper's traceability discussion.

## Controls file schema

Any JSON or JSONL where each entry has a text field. Recognised aliases:

| Field       | Aliases                        |
|-------------|--------------------------------|
| `control_id`| `id`                           |
| `text`      | `control_text`, `body`         |
| `source`    | `framework`                    |
| `section`   | `section_id`                   |

Unknown fields are preserved under `metadata`. If the JSON is a wrapper
object, the loader looks for `controls`, `entries`, `items`, or `data` as
the list key.

## Reproducibility

- Cosine similarity, argpartition + stable sort → deterministic top-k
- Encoder normalisation and float32 storage → bit-exact index across runs
- Corpus hash written into `index_metadata.json` → detect stale rebuilds
- Random state / seed forwarded through pipeline unchanged
