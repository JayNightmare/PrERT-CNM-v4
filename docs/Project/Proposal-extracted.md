# Proposal Extraction

Source: `docs/Project/Proposal.pdf`

## Aim

Develop AI methods to quantify user privacy risks in alignment with international standards and regulations (ISO/IEC, NIST, GDPR, and related frameworks), addressing gaps in existing tools that:

- Analyze privacy policies but do not integrate standards-based compliance measurement.
- Lack measurable and interoperable privacy-risk indicators.

Planned outcomes include:

- Validated privacy metrics.
- A proof-of-concept tool combining PrivacyBERT-based clause analysis with Bayesian/probabilistic risk modeling.
- A roadmap toward a broader standards-aligned privacy risk quantification framework.

## Approach

The proposal defines a four-step delivery approach:

1. Map measurable privacy principles from ISO/IEC, NIST, GDPR, IEEE, and related regulations into quantifiable indicators (e.g., consent, data minimization, encryption, third-party sharing).
2. Design and test privacy-risk metrics across user, system, and organizational levels using synthetic logs and public breach datasets (ENISA, PRC), with partial implementation for organization-level indicators and conceptual scoping for digital ecosystem issues.
3. Develop an AI prototype that combines:
      - PrivacyBERT fine-tuned on OPP-115 and Polisis datasets for clause classification.
      - Bayesian/probabilistic models for privacy-risk scoring.
4. Validate and refine the prototype by benchmarking against defined metrics, then deliver the validated tool and final report.

## Phase Requirements (from Timetable / Plan)

### Phase 1 (Month 1)

Requirements:

- Map measurable privacy principles from ISO/IEC, NIST, GDPR, IEEE, and other regulations into privacy indicators.

Deliverable:

- International standards to privacy metrics mapping.

### Phase 2 (Month 2)

Requirements:

- Define and test user-, system-, and organization-level privacy metrics.
- Scope digital ecosystem-level privacy indicators.
- Generate synthetic datasets for testing.

Deliverable:

- Draft privacy metrics with synthetic data.

### Phase 3 (Month 3)

Requirements:

- Build AI prototype for privacy clause classification using PrivacyBERT.
- Integrate Bayesian risk models for privacy-risk scoring.

Deliverable:

- Prototype user privacy quantification AI tool.

### Phase 4 (Month 4)

Requirements:

- Test prototype on real and synthetic data.
- Benchmark against defined metrics.
- Finalize reporting outputs.

Deliverable:

- Validated tool and final report.
