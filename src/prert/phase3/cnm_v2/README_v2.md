# CNMv2 — Auxiliary-head retrieval architecture

Redesign of the Contextual Neural Memory to avoid the class-discrimination collapse observed in CNMv1's naive text-concatenation approach.

## Architecture

```mermaid
    clause x
        │
        ▼
    PrivBERT encoder  ────► h(x) ∈ ℝ^H         (no retrieval concatenation)
        │                       │
        ├─ text head ──────────► logits_T = W_T · h(x)          ∈ ℝ^C
        │                       │
        ├─ retrieval head ─────► logits_R = Σ_j softmax(sim/τ)_j · L[r_j]
        │                       │  where L ∈ ℝ^{|M|×C} is per-control learned logits
        │                       │
        └─ gate ───────────────► α = σ(w_g · h(x) + b_g)  ∈ [0,1]
                                │
    logits = α · logits_T + (1-α) · logits_R
    p = softmax(logits)
```

**Key differences from CNMv1:**

| | CNMv1 (concatenation) | CNMv2 (auxiliary head) |
| - | - | - |
| Encoder input | `clause [SEP] c_1 [SEP] ... [SEP] c_k` | raw `clause` only |
| Retrieval effect | Dilutes clause signal in encoder input | Adds a separate logit stream |
| Failure mode | Minority-class collapse (observed at k=1) | Gate falls back to text head |
| Baseline recovery | Impossible (input distribution shifted) | α → 1 recovers baseline exactly |
| Traceability | Retrieved controls attached to prediction | Same, plus per-control learned class-prior |

**Safety properties (verified in tests):**

1. **Gate init biases toward text head.** At step 0, `α ≈ 0.88`, so the model behaves ~identically to the plain PrivBERT baseline.
2. **Control logits initialised to zero.** The retrieval head contributes zero signal at step 0.
3. **Freeze-below-epoch option.** `--freeze-retrieval-below-epoch 1` keeps retrieval + gate frozen during the first epoch, letting the text head converge to baseline first.

## Files

| File | Purpose |
| ------ | --------- |
| `model.py` | `CNMv2Model` (nn.Module), `CNMv2Config`, `CNMv2Output` |
| `memory_chroma.py` | `ChromaMemoryIndex`, `ControlRecord` — Chroma-backed catalogue |
| `classifier_v2.py` | `CNMv2Classifier` + custom `Trainer` with dual loss |
| `pipeline_v2.py` | `run_phase3_cnmv2_pipeline`, `run_cnmv2_ablation` |
| `cli_v2.py` | `prert-cnmv2 {build-chroma,phase3,ablation}` |
| `tests/test_model_v2.py` | 7 architectural invariant tests |

## Running

### 1. Build the ChromaDB index (one-time, ~2 min)

```bash
python -m prert.phase3.cnm_v2.cli_v2 build-chroma \
    --controls-path artifacts/phase-1/controls.jsonl \
    --output-dir artifacts/cnm-index-v2
```

Verify it built cleanly:

```bash
ls artifacts/cnm-index-v2/                 # should show chroma/ and index_metadata.json
cat artifacts/cnm-index-v2/index_metadata.json    # count should be 956 (or your control count)
```

### 2. Test at k=5 first before running the full sweep

```bash
python -m prert.phase3.cnm_v2.cli_v2 phase3 \
    --cnm-index artifacts/cnm-index-v2 \
    --top-k 5 \
    --output-dir artifacts/phase-3-cnmv2-k5 \
    --run-id cnmv2-k5 \
    --privacybert-epochs 3 \
    --freeze-retrieval-below-epoch 1
```

Expected: `model_type=cnmv2_privacybert`, test macro-F1 ≥ 0.895 (should meet or beat baseline).

If macro-F1 comes back **below 0.85**, stop — the gate has failed to fall back to text-head-only mode and something's off. If it comes back **at 0.895 exactly**, the retrieval head is contributing nothing (gate stuck at α=1). If it comes back at **0.90-0.92**, we're winning.

### 3. Full ablation (only if the k=5 pilot works)

```bash
python -m prert.phase3.cnm_v2.cli_v2 ablation \
    --cnm-index artifacts/cnm-index-v2 \
    --k-values 0,1,3,5,10 \
    --output-dir artifacts/phase-3-cnmv2-ablation \
    --run-id cnmv2-ablation \
    --privacybert-epochs 3 \
    --freeze-retrieval-below-epoch 1
```

Runtime estimate: ~3 hours per k × 5 = **~15 hours**. Similar to CNMv1's ablation.

## Diagnostics

Every training run logs:

```log
Training CNMv2Classifier (top_k=5, index=..., freeze_below_epoch=1)
Precomputing top-5 retrievals for 15484 clauses (memory=956)
...training log lines...
```

If you don't see `Precomputing top-5 retrievals`, the retrieval precompute failed silently.

Every prediction with `predict_with_trace()` returns:

```json
{
  "probabilities": {"user": 0.02, "system": 0.11, "organization": 0.87},
  "gate_alpha": 0.73,
  "retrieved": [
    {"control_id": "GDPR-Art5", "source": "GDPR", "score": 0.81, "text": "..."},
    ...
  ]
}
```

The `gate_alpha` value tells you how much the model trusted retrieval for that specific clause: closer to 1 = mostly text head, closer to 0 = mostly retrieval head. This is the paper's headline explainability signal.

## Segfault mitigation

CNMv1's ablation crashed with a segfault between k=1 and k=3. To avoid that in CNMv2:

1. **Run each k separately** rather than a single ablation command:

   ```bash
   for k in 0 1 3 5 10; do
     python -m prert.phase3.cnm_v2.cli_v2 phase3 --top-k $k --output-dir artifacts/phase-3-cnmv2-ablation/k=$k ...
   done
   ```

2. The pipeline_v2 shim restores `train_classifier` bindings in a `finally` block, so no state leaks between k values.

## Test verification

```bash
cd D:\Documents\Personal-Projects\PrERT-CNM-v4
python -m pytest tests/test_model_v2.py -v
```

Expected: `7 passed`.

## What CNMv2 will NOT do

- **Beat 0.98 macro-F1.** OPP-115 is at ceiling around 0.90; the practical improvement zone is +0.02 to +0.05 over baseline, driven by minority-class recovery. Anything above +0.05 on the same split would be a red flag worth investigating.
- **Improve results without retrieval consistency.** If `--retrieval-consistency-weight 0`, the per-control logits are unregularised and may not converge to anything useful. Keep the default 0.1.
- **Replace the traceability story.** The primary contribution remains "every prediction cites specific controls". CNMv2 just adds a mechanism whereby those controls can also *contribute to* the prediction, not merely explain it.
