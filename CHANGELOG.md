# Changelog

## [0.7.0] - 2026-07-09

### Added

- **Live PrivacyBERT benchmarking workflow** with artifact handling improvements for Phase 4 validation runs.
- **Primary model identification** and **Hugging Face release guidance** integrated into the live benchmarking flow.
- **Synthetic policy and document classes** to strengthen Phase 4 synthetic schema generation coverage.
- **MSc thesis documentation package** (thesis source, references, and formative presentation assets) aligned to the PrERT-CNM-v4 methodology and results.

### Changed

- **Compliance pipeline update**: Bayesian risk assessment is now integrated more directly into the compliance flow, and legacy artifacts were pruned.
- **Maintainability refactor** across core modules to improve readability and long-term extensibility.
- **Documentation and artifact formatting refresh** across Phase 2/3/4 supporting assets for consistency.

### Commits

- `277181a` Add master thesis document and references for MSc project on privacy-cybersecurity intelligence model
- `5b00c43` Add main LaTeX document and references for PrERT-CNM-v4 framework
- `438625f` chore: update totals
- `84fc254` chore: update totals
- `24949e4` feat: Enhance live benchmarking with primary model identification and Hugging Face release instructions
- `e3d4153` feat: Implement live PrivacyBERT benchmarking and enhance artifact management
- `0849838` Refactor code structure for improved readability and maintainability
- `c9e6cbd` style: Format markdown files for consistency and readability across phase 2 and phase 3 artifacts
- `f7681f6` chore: update totals
- `192d971` Add main.tex and references.bib for PrERT-CNM-v4 documentation
- `a4f41c4` feat: Add synthetic policy and document classes, enhance phase 4 schema generation
- `875e136` refactor: integrate bayesian risk assessment into compliance pipeline and prune legacy artifact files
- `7322a20` Refactor test_phase3_pipeline.py: Reorganize imports and improve dataset writing functions for clarity and maintainability

## [0.6.0] - 2026-06-22

### Added

- **Policy-only compliance assessment** (`assess_policy_compliance`): evaluates a privacy policy against GDPR, NIST Privacy Framework, and ISO/IEC 27701 without requiring a database schema.
- **Source citations** for compliance verdicts: each pass/fail decision cites policy clause text.
- **Per-regulation independent scoring** via structured `RegulationVerdict` objects (GDPR articles, NIST subcategories, ISO 27701 controls).
- **`REGULATION_CONTROLS` mapping** across the policy check areas for deterministic, framework-level checks.
- **Comprehensive policy-only tests** for coverage, citation integrity, and grading consistency.

### Changed

- **Test pipeline refactor** for `test_phase3_pipeline.py` to improve readability and dataset writing clarity.

### Commits

- `3116d1d` feat: implement policy-only compliance assessment with multi-regulation verification and comprehensive test suite

## [0.5.0] - 2026-05-28

### Added

- **Phase 4 validation reports** and **APP-350 processing workflow** for expanded validation outputs.
- **PrivacyBERT training enhancements** with additional loss-function options and training parameters.
- **Synthetic policy/schema generation improvements** with progress callback support.
- **Project proposal and references** to formalize research documentation.

### Changed

- **ISO/GDPR/NIST parsing improvements** with stronger metrics handling and data explorer support.
- **Codebase readability refactors** across selected modules.

### Fixed

- **Python version requirements** corrected in documentation and packaging metadata (`README` and `pyproject.toml`) to 3.11-3.14.

### Commits

- `96a62f1` Refactor code structure for improved readability and maintainability
- `df4f82c` feat: Enhance PrivacyBert Classifier with additional loss functions and training parameters
- `3d98328` fix: update Python version requirements to 3.11-3.14 in README and pyproject.toml
- `80c4753` Add Phase 4 validation reports and APP-350 processing functionality
- `355bb8f` Enhance ISO and NIST parsers, improve metrics handling, and add data explorer functionality
- `c0c205e` chore: update totals
- `cefc15c` Add MSc project proposal and references for user privacy risk quantification
- `3a3de61` feat: Enhance synthetic policy schema dataset generation with progress callbacks
- `934e112` feat: Enhance document parsing for ISO and GDPR standards

## [0.4.0] - 2026-04-14

### Added

- **Streamlit GUI** for policy + schema compliance assessment.
- **Phase 4 demo bundle script and manifest** to support reproducible demonstrations.
- **Polisis evidence requirement** in Phase 4 validation, with corresponding README and test updates.
- **Phase 1-3 critical review documentation** for proposal-alignment assessment.

### Changed

- **Phase 3 pipeline enhancements** with calibration and bootstrap analytics.
- **README and WakaTime total reporting format** updates.
- **Project packaging adjustments** in `pyproject.toml` and `.gitignore` updates for generated app files.

### Fixed

- **Command compatibility fix** replacing `python3` with `python` in scripts and documentation.

### Removed

- **Tracked artifacts directory contents** removed from source control and ignored via `.gitignore` to reduce repository bloat.

### Commits

- `d83ecf7` chore: update totals
- `7360226` feat: Update README totals format and enhance WakaTime total calculation script
- `35b0361` feat: Add requirement for Polisis source evidence in Phase 4 validation and update README and tests
- `40b522f` feat: Update .gitignore to include Streamlit text files and modify pyproject.toml for poetry package configuration
- `bbbd2dc` Add Phase 4 demo bundle preparation script and manifest
- `9d63000` feat: Add Streamlit GUI for policy and schema compliance assessment
- `af4a5ed` feat: Add Phase 1-3 Critical Review document for proposal alignment assessment

## [0.3.0] - 2026-04-07

### Added

- **Phase 3 acceptance freeze workflow** with associated evaluation artifacts.
- **PrivacyBERT integration** and **Bayesian scoring** in the Phase 3 path.

### Changed

- **Phase 3 baseline classifier pipeline** matured with evaluation and visualization support.
- **Documentation expansion** for Phase 2 and Phase 3 implementation details.
- **Readability and maintainability refactors** across core project modules.

### Fixed

- **Repository hygiene fix** removing tracked artifacts and moving them to ignore rules for cleaner source control behavior.
- **Execution command compatibility fix** replacing `python3` with `python` across scripts and docs.

### Commits

- `9a1e91b` Refactor code structure for improved readability and maintainability
- `10dcec0` feat: Enhance Phase 3 pipeline with calibration and bootstrap analytics
- `12f491d` Refactor code structure for improved readability and maintainability
- `934d63b` fix: Replace python3 with python in execution commands across documentation and scripts
- `438de3f` chore: Add artifacts directory to .gitignore
- `cb46200` Delete artifacts directory
- `57ae80d` docs: Update README to include detailed descriptions of Phase 2 and Phase 3 implementations
- `5a08b01` Add Phase 3 acceptance freeze workflow and evaluation
- `8e22b42` feat: Enhance Phase 3 with PrivacyBERT integration and Bayesian scoring
- `f9b7418` Refactor code structure for improved readability and maintainability
- `a02042b` refine: Update Phase 3 visual dashboard metrics for clarity and consistency
- `8f075ad` feat: Implement Phase 3 baseline classifier pipeline with evaluation and visualization

## [0.2.0] - 2026-04-05

### Added

- **Phase 1/2 dashboard generation script** and visual snapshots for progress reporting.
- **Implementation runbooks and technical documentation** for early phase execution.
- **Enhanced OPP-115 processing support** and improved document index coverage.

### Changed

- **Phase 1-2 dashboard clarity improvements** (legend, descriptions, metrics formatting, and contextual presentation).
- **README visualization coverage** expanded.

### Commits

- `22ef55a` Format compliance and risk metrics in Phase 1-2 Progress Dashboard for improved readability
- `3aaa991` Refine Phase 1-2 Progress Dashboard legend and figure descriptions for clarity and context
- `38079ed` Enhance Phase 1-2 Progress Dashboard with additional context and visual clarity
- `d02d318` Add script to generate Phase 1/2 dashboard figures from artifact data
- `cf59be8` Update README to include visualizations and enhance document index
- `0128df3` Add Phase 1 and Phase 2 visual snapshots, enhance OPP-115 processing, and update documentation
- `63ad9f2` Organize implementation runbooks and technical documentation for Phase 1 and Phase 2

## [0.1.0] - 2026-04-05

### Added

- **Initial end-to-end project scaffold** spanning extraction, chunking, Phase 2 metrics, Phase 3 baseline, Phase 4 foundations, scripts, tests, and documentation.
- **Baseline artifacts and repository structure** for PrERT-CNM-v4 research workflows.

### Commits

- `e3f014c` first commit

## [0.0.1] - 2026-04-05

### Added

- **Historical baseline marker** for repository origin prior to structured milestone tracking.

### Commits

- `e3f014c` first commit (shared bootstrap commit used as baseline provenance)
