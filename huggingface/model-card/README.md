---
language: en
library_name: transformers
pipeline_tag: text-classification
tags:
- bert
- privacy
- text-classification
- gradio
model-index:
- name: {{MODEL_ID}}
  results: []
---

# {{MODEL_ID}}

PrERT-CNM v4 PrivacyBERT is a Transformers sequence-classification model prepared from the local checkpoint at `{{CHECKPOINT_PATH}}`.

## Intended Use

Use this model for text classification in the privacy/CNM workflow it was trained for. It is intended for research and application prototyping unless your own validation shows it is suitable for production use.

## Labels

{{LABELS}}

## Usage

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="{{MODEL_ID}}", top_k=None)
scores = classifier("Paste text to classify.")
print(scores)
```

## Training Details

- Base architecture: BERT-compatible sequence classifier
- Source checkpoint: `{{CHECKPOINT_PATH}}`
- Training metadata: included when available in the checkpoint folder

## Evaluation

Add the final held-out metrics before publishing if they are available. Include dataset split details, label distribution, and any thresholding used by downstream consumers.

## Limitations

The model can be sensitive to domain shift, ambiguous language, long inputs, and label definitions that differ from the training data. Review outputs before using them in automated decisions.

## Gradio Demo

The companion Space can be prepared from `huggingface/space` and pointed at this model with the `MODEL_ID` environment variable.
