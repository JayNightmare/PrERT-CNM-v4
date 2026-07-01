#!/usr/bin/env python3
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "JayNightmare/PrERT-CNM-v4-privacybert"
DEFAULT_CHECKPOINT = ROOT / "artifacts/phase-3-privacybert/classifier_checkpoint/privacybert"
MODEL_TEMPLATE_DIR = ROOT / "huggingface/model-card"
SPACE_TEMPLATE_DIR = ROOT / "huggingface/space"
PACKAGE_SOURCE_DIR = ROOT / "src/prert"
PHASE1_CONTROLS_SOURCE = ROOT / "artifacts/phase-1/controls_all.jsonl"

MODEL_FILE_NAMES = {
    ".gitattributes",
    "added_tokens.json",
    "config.json",
    "flax_model.msgpack",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "spiece.model",
    "tf_model.h5",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
    "training_metadata.json",
    "vocab.json",
    "vocab.txt",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Hugging Face model and Space upload folders.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-output", type=Path, default=ROOT / "dist/huggingface/model")
    parser.add_argument("--space-output", type=Path, default=ROOT / "dist/huggingface/space")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--space-title", default="PrERT-CNM Compliance Studio")
    parser.add_argument("--clean", action="store_true", help="Remove output folders before writing.")
    return parser.parse_args()


def read_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_clean_dir(path, clean=False):
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_named_files(source_dir, target_dir, allowed_names):
    copied_files = []
    for source_file in sorted(source_dir.iterdir()):
        if source_file.is_file() and source_file.name in allowed_names:
            target_file = target_dir / source_file.name
            shutil.copy2(source_file, target_file)
            copied_files.append(target_file.name)
    return copied_files


def labels_from_metadata(config, metadata):
    id_to_label = config.get("id2label") or metadata.get("id2label")
    if isinstance(id_to_label, dict):
        labels = [label for _, label in sorted(id_to_label.items(), key=lambda item: int(item[0]))]
        return "\n".join(f"- `{label}`" for label in labels)

    label_to_id = config.get("label2id") or metadata.get("label2id")
    if isinstance(label_to_id, dict):
        labels = [label for label, _ in sorted(label_to_id.items(), key=lambda item: int(item[1]))]
        return "\n".join(f"- `{label}`" for label in labels)

    labels = metadata.get("labels") or metadata.get("class_names")
    if isinstance(labels, list):
        return "\n".join(f"- `{label}`" for label in labels)

    return "Add the final label names before publishing if they are not already stored in `config.json`."


def render_model_card(checkpoint, model_output, model_id):
    config = read_json(checkpoint / "config.json")
    metadata = read_json(checkpoint / "training_metadata.json")
    template = (MODEL_TEMPLATE_DIR / "README.md").read_text(encoding="utf-8")
    try:
        checkpoint_path = str(checkpoint.relative_to(ROOT))
    except ValueError:
        checkpoint_path = str(checkpoint)

    rendered = template.replace("{{MODEL_ID}}", model_id)
    rendered = rendered.replace("{{CHECKPOINT_PATH}}", checkpoint_path)
    rendered = rendered.replace("{{LABELS}}", labels_from_metadata(config, metadata))
    (model_output / "README.md").write_text(rendered, encoding="utf-8")


def copy_model_card_assets(model_output):
    for template_file in MODEL_TEMPLATE_DIR.iterdir():
        if template_file.name == "README.md" or not template_file.is_file():
            continue
        shutil.copy2(template_file, model_output / template_file.name)


def prepare_space(space_output, model_id, space_title):
    for source_file in sorted(SPACE_TEMPLATE_DIR.iterdir()):
        if not source_file.is_file():
            continue
        target_file = space_output / source_file.name
        text = source_file.read_text(encoding="utf-8")
        text = text.replace("__MODEL_ID__", model_id)
        text = text.replace("__SPACE_TITLE__", space_title)
        target_file.write_text(text, encoding="utf-8")

    package_target = space_output / "src/prert"
    if package_target.exists():
        shutil.rmtree(package_target)
    shutil.copytree(
        PACKAGE_SOURCE_DIR,
        package_target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )

    # Include ground-truth controls corpus so the Space can list all frameworks.
    if PHASE1_CONTROLS_SOURCE.exists():
        controls_target_dir = space_output / "artifacts/phase-1"
        controls_target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PHASE1_CONTROLS_SOURCE, controls_target_dir / "controls_all.jsonl")


def validation_warnings(checkpoint, copied_files):
    warnings = []
    if not checkpoint.exists():
        warnings.append(f"Checkpoint folder does not exist: {checkpoint}")
    if not {"model.safetensors", "pytorch_model.bin", "tf_model.h5", "flax_model.msgpack"}.intersection(copied_files):
        warnings.append("No model weight file was copied. Expected one of model.safetensors, pytorch_model.bin, tf_model.h5, or flax_model.msgpack.")
    if "config.json" not in copied_files:
        warnings.append("config.json was not copied. Transformers needs it to load the model.")
    if not {"tokenizer.json", "vocab.txt", "vocab.json", "spiece.model"}.intersection(copied_files):
        warnings.append("No tokenizer vocabulary file was copied. Confirm the tokenizer files are present in the checkpoint folder.")
    return warnings


def write_manifest(model_output, space_output, checkpoint, model_id, copied_files, warnings):
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "checkpoint": str(checkpoint),
        "model_output": str(model_output),
        "space_output": str(space_output),
        "copied_model_files": copied_files,
        "warnings": warnings,
    }
    (model_output / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    model_output = args.model_output.resolve()
    space_output = args.space_output.resolve()

    ensure_clean_dir(model_output, clean=args.clean)
    ensure_clean_dir(space_output, clean=args.clean)

    copied_files = copy_named_files(checkpoint, model_output, MODEL_FILE_NAMES) if checkpoint.exists() else []
    copy_model_card_assets(model_output)
    render_model_card(checkpoint, model_output, args.model_id)
    prepare_space(space_output, args.model_id, args.space_title)

    warnings = validation_warnings(checkpoint, copied_files)
    write_manifest(model_output, space_output, checkpoint, args.model_id, copied_files, warnings)

    print(f"Model upload folder: {model_output}")
    print(f"Space upload folder: {space_output}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()