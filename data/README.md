# Dataset Training and Validation

> [!IMPORTANT]
> Ensure you have `raw/` and `processed/`. Raw should contain the OPP-115 dataset extracted from the zip, while processed datasets should be built through the Phase 3 workflow documented in [docs/Project/Execution-Playbook/10-phase3-implementation-runbook.md](../docs/Project/Execution-Playbook/10-phase3-implementation-runbook.md).

The raw OPP-115 dataset can be found on [Usable Privacy](https://www.usableprivacy.org/static/data/OPP-115_v1_0.zip). Follow the build steps in [docs/Project/Execution-Playbook/10-phase3-implementation-runbook.md](../docs/Project/Execution-Playbook/10-phase3-implementation-runbook.md) to create the processed datasets used by the default Phase 3 training and evaluation flow.

At present, OPP-115 remains the canonical supervised dataset for the committed Phase 3 benchmark artefacts. Polisis, external privacy-policy corpora, and synthetic augmentation are research extensions under active evaluation rather than the default reproducible path.

The raw dataset is retained for reference and traceability. The processed datasets are the inputs used for Phase 3 training and held-out validation, and they also feed the downstream Phase 4 comparison workflow.
