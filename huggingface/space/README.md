---
title: __SPACE_TITLE__
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
python_version: 3.11
pinned: false
---

Default model: `__MODEL_ID__`.

Set `MODEL_ID` to override the model repository at runtime.

This Space provides policy-only compliance assessment, hosted synthetic policy/schema generation with downloadable outputs, stored model benchmark comparison, and a raw classifier view.

## Compliance Demo Features

- Real-time stage progression during claim parsing, check matching, and verdict scoring.
- Evidence-link graph that grows live from claim -> control -> framework connections.
- Framework selector to evaluate all discovered regulations or a focused subset.
- Evidence-backed rationale strings with explicit source and match score fields.

The graph is a visualization of evidence-link formation and scoring flow. It is not a direct view of hidden neural activations.

## Chroma Retrieval Configuration

To enable remote retrieval from Chroma in this Space, set these secrets/variables:

- `CHROMA_HOST`
- `CHROMA_API_KEY`
- `CHROMA_TENANT`
- `CHROMA_DATABASE`
- `CHROMA_COLLECTION_NAME` (optional; defaults to `ground_truth`)

If these values are not provided or connection fails, the app falls back to local ground-truth retrieval from `artifacts/phase-1/controls_all.jsonl`.

## Optional Environment Variables

- `MODEL_ID`: override the model repository used by the classifier tab.
- `MODEL_REVISION`: model revision/tag/branch.
- `MAX_LENGTH`: classifier max sequence length.
- `PRERT_CONTROLS_PATH`: optional custom controls JSONL path for local retrieval.
