# Dataset Options Paper Trail

This note records the current dataset strategy for PrERT-CNM-v4 against the project aim of improving accuracy credibly before any publication attempt.

## Verified Current Position

- The current supervised benchmark anchor in the repository is OPP-115.
- The committed benchmark artefacts are centred on OPP-115-derived Phase 3 runs and the `phase-3-freeze` baseline.
- The live Phase 3 code already supports OPP-115, normalised Polisis inputs, and manual labelled JSONL inputs.
- The current label surface in code is `user`, `system`, and `organization`; any additional dataset must either map cleanly into that space or be used for a different purpose such as pretraining, weak supervision, or evaluation.
- Synthetic data remains in scope, but it should be treated as a documented augmentation or challenge-set asset rather than an untracked substitute for benchmark data.
- A controlled APP-350 auxiliary experiment has now been run and should be treated as negative evidence for the current label space: the conservative mapping retained zero `user` rows and degraded the OPP-only benchmark.

## Recommendation Tiers

### Tier 1: Use Now

These datasets or assets directly support the current benchmark and defensibility path.

| Dataset / Asset                           | Best Use                                                          | Label Fit                                                          | Licence / Risk Notes                                                                        | Ingestion Cost | Recommendation                                                                             |
| ----------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------ |
| OPP-115 Corpus                            | Primary supervised benchmark anchor                               | High via existing repo mapping                                     | Research, teaching, and scholarship use only; non-commercial style terms                    | Low            | Keep as the main benchmark anchor until a stronger alternative is proven                   |
| OPP-115 to GDPR links (JURIX 2020)        | Standards alignment and defensible mapping notes                  | Medium as an auxiliary mapping asset rather than a training set    | Citation and provenance required; use as supporting evidence rather than replacement labels | Low            | Use to strengthen standards-traceable interpretation without changing the benchmark anchor |
| Manually reviewed synthetic challenge set | Targeted gap analysis for underrepresented privacy-rights clauses | High if written to the existing `user/system/organization` surface | Must be explicitly documented, reviewable, and clearly separated from benchmark claims      | Medium         | Use for challenge evaluation and carefully controlled augmentation                         |

### Tier 2: Next Candidate Ingestion

These are the strongest near-term candidates for improving the privacy angle, but they require label harmonisation and explicit provenance work.

| Dataset / Asset                   | Best Use                                                                                     | Label Fit                                                                                                   | Licence / Risk Notes                                                                    | Ingestion Cost | Recommendation                                                                                                       |
| --------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------- |
| Polisis-derived normalised corpus | Controlled augmentation and out-of-domain evaluation                                         | Medium; category harmonisation already exists in code but still needs curated provenance and deduplication  | Source availability and reuse terms must be confirmed for the exact corpus variant used | Medium         | Prioritise after OPP-115 because the repo already has a loader path and category mapping                             |
| APP-350 Corpus                    | Auxiliary supervised signal for privacy-practice language in app policies                    | Poor in the current conservative mapping because retained coverage collapses to `organization` and `system` | Research, teaching, and scholarship use only; non-commercial style terms                | Medium         | Do not prioritise further for the current Phase 3 label space unless a new mapping yields defensible `user` coverage |
| MAPP Corpus                       | More recent privacy-policy augmentation and evaluation, especially around regulatory framing | Medium; bilingual corpus means English-only filtering or bilingual handling is needed                       | Research, teaching, and scholarship use only; non-commercial style terms                | Medium to high | Good candidate for a second ingestion wave after APP-350 if mobile-policy and post-GDPR phrasing matter              |
| Opt-out Choice Dataset            | Targeted augmentation for user-choice, consent, and opt-out language                         | Medium to low for the full label space, but high for the `user` minority class                              | Research, teaching, and scholarship use only; non-commercial style terms                | Medium         | Use as a targeted auxiliary asset rather than a full replacement dataset                                             |

### Tier 3: Domain Adaptation, Weak Supervision, or Evaluation Only

These datasets are valuable, but they are not natural drop-in replacements for the current benchmark.

| Dataset / Asset                                          | Best Use                                                                                 | Label Fit                                                                                                         | Licence / Risk Notes                                                                                                          | Ingestion Cost | Recommendation                                                                                            |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | -------------- | --------------------------------------------------------------------------------------------------------- |
| PrivaSeer Corpus                                         | Domain-adaptive pretraining, corpus mining, retrieval, or weak supervision               | Low for direct supervised use in the current label space                                                          | CC BY-NC-SA for research, teaching, and scholarship; non-commercial restriction remains relevant                              | High           | Use for domain adaptation or candidate-sentence mining, not as the main benchmark by itself               |
| Princeton-Leuven Longitudinal Corpus of Privacy Policies | Temporal robustness checks and evaluation on policy drift over time                      | Low for direct supervised use because it is a historical corpus rather than a label-aligned clause dataset        | Public GitHub access is available, but publication-time licence and downstream reuse terms should still be checked explicitly | High           | Use later for external validity and robustness rather than immediate model training                       |
| MAPS Policies Dataset                                    | Large-scale unlabelled privacy-policy discovery for weak supervision or corpus expansion | Low as a direct training signal because it provides policy locations rather than label-aligned clause annotations | Research, teaching, and scholarship use only; non-commercial style terms                                                      | High           | Only worthwhile if the project moves into larger-scale weak supervision or crawler-backed corpus building |
| PrivacyQA                                                | Question-answering evaluation for privacy-policy comprehension                           | Low for clause classification                                                                                     | GitHub-distributed corpus; task does not match the repo’s current classifier target directly                                  | Medium         | Useful for adjacent evaluation only, not for the main classification objective                            |

## Label-Fit Notes

The current Phase 3 task is not a direct reproduction of the original category sets used by all privacy-policy corpora. The repo maps source categories into `user`, `system`, and `organization`.

That means:

- OPP-115 is the cleanest benchmark anchor because this mapping is already implemented and evidenced in the current repo.
- Polisis-derived and APP-350-style corpora should only be used after writing down the mapping rules, skipped categories, and any dropped rows.
- Large corpora such as PrivaSeer or Princeton-Leuven are better treated as domain-adaptation or evaluation resources unless a defensible labelling workflow is added.

## Licence and Publication Cautions

- OPP-115, APP-350, MAPP, and the opt-out datasets are released for research, teaching, and scholarship use, with non-commercial style restrictions noted by the source project.
- PrivaSeer is available under CC BY-NC-SA for research, teaching, and scholarship purposes.
- For any dataset used in a publication attempt, keep the original citation, source URL, access date, and any processing exclusions in the repo.
- For datasets whose downstream redistribution terms are unclear in the exact form used, store only manifests and transformation scripts in the repo rather than republishing the raw source content.

## Recommended Immediate Strategy

1. Keep OPP-115 as the benchmark anchor for all near-term accuracy claims.
2. If auxiliary-data work continues, prioritise Polisis-derived normalised data only after clean provenance is secured; APP-350 should remain paused unless a materially better label mapping is justified.
3. Use synthetic data only where it closes an identifiable gap, especially user-rights, opt-out, retention, deletion, and third-party-sharing language.
4. Treat PrivaSeer and Princeton-Leuven as medium-term assets for domain adaptation and external validation rather than immediate benchmark replacement.

## Recommended Evidence to Record for Any New Dataset

For every new dataset brought into the repo workflow, record:

- source URL and citation
- licence or reuse restriction summary
- local storage path
- transformation script or normalisation step
- label-mapping rules
- dropped rows and exclusion reasons
- duplicate-policy checks against OPP-115 and other incorporated corpora
- exact experiment runs that used the dataset

That provenance record is required if the project is to remain academically defensible and publication-oriented.
