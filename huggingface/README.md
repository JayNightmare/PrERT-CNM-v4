# Hugging Face Release Kit

This folder contains the files needed to publish the PrivacyBERT checkpoint as a Hugging Face model repository and a Gradio Space for the Phase 4 compliance dashboard.

## Build the Upload Directories

From the repository root:

```bash
python scripts/prepare_huggingface_release.py \
  --checkpoint artifacts/phase-3-privacybert/classifier_checkpoint/privacybert \
  --model-output dist/huggingface/model \
  --space-output dist/huggingface/space \
  --model-id JayNightmare/PrERT-CNM-v4-privacybert \
  --space-title "PrERT-CNM Compliance Studio"
```

The script copies the Transformers checkpoint files, adds a model card, writes a release manifest, and prepares the Gradio Space with the selected model id, bundled Phase 4 source package, and stored benchmark registry.

## Local Smoke Test

Install the Space dependencies and point the app at the generated model folder:

```bash
python -m pip install -r huggingface/space/requirements.txt
MODEL_ID=dist/huggingface/model python huggingface/space/app.py
```

The Space includes policy-only compliance assessment, hosted synthetic data generation with copyable samples and downloads, stored model benchmark comparison, artifact browsing, and a raw PrivacyBERT classifier tab.

The Synthetic Data tab writes temporary Space-managed files and exposes the generated policy/schema samples, dataset JSONL, manifest JSON, and upload-fixture ZIP directly in the UI. It does not require a local output directory or model checkpoint path.

The Benchmark Validation tab displays bundled published metrics from `huggingface/space/benchmarks.json`; it does not require artifact directories.

## Upload

Create the repositories on Hugging Face, then upload the generated folders:

```bash
hf auth login
hf upload JayNightmare/PrERT-CNM-v4-privacybert dist/huggingface/model --repo-type model
hf upload JayNightmare/PrERT-CNM-v4-privacybert-demo dist/huggingface/space --repo-type space
```

If the model repository is private, add `HF_TOKEN` as a Space secret so the Space can load it.
