# **Ethical Integrity and Data Governance in AI-Driven Privacy Quantification: A Synthesis of the PrERT-CNM Framework**

The rapid advancement of transformer-based models in the field of natural language processing (NLP) has created a significant gap between the technical capability to classify privacy policy text and the institutional requirement for explainable, standards-aligned risk reasoning \[cite: 1\]. Within the PrERT-CNM project, the "Ethics & Data Governance" framework serves as a foundational architecture to bridge this divide, ensuring that the quantification of user privacy risks remains both legally compliant and technically robust \[cite: 1\]. By centering the research methodology on five core pillars—non-participatory collection, academic licensing compliance, the pairing of synthetic and public corpora, the prioritization of professional judgment over probabilistic outputs, and the enforcement of human-in-the-loop oversight—the project establishes a high-level ethical posture that aligns with international regulatory standards including the General Data Protection Regulation (GDPR), various ISO/IEC frameworks, and the NIST AI Risk Management Framework \[cite: 1, 2, 3\].

# **Non-Participatory Collection and the Principle of Data Minimization**

A primary ethical claim of the PrERT-CNM pipeline is the strict avoidance of direct personal data collection from human participants \[cite: 1, 4\]. This methodological choice is a critical safeguard that distinguishes the project from traditional behavioral or medical AI research, which often requires extensive Institutional Review Board (IRB) oversight and carries inherent risks regarding participant vulnerability and mental autonomy \[cite: 5, 6\]. By design, the project bypasses the need for human participant recruitment, focusing instead on the analysis of public-facing legal documents \[cite: 1, 4, 5\].

## **Regulatory Derogations for Scientific Research**

This non-participatory approach is deeply rooted in the concept of "Processing which does not require identification" as outlined in Article 11 of the GDPR \[cite: 7\]. Under this framework, if the purposes for which a controller processes data do not require the identification of a data subject, the controller is not obliged to maintain or acquire additional information solely for compliance \[cite: 7\]. The PrERT-CNM pipeline adheres to this by processing only segment-level annotations from established corpora, ensuring that the "need-to-know" principle is satisfied throughout the data lifecycle \[cite: 8\].

Furthermore, Article 89 of the GDPR provides specific derogations for scientific research purposes, provided that technical and organizational measures are in place to ensure data minimization \[cite: 7, 9\]. In the context of PrERT-CNM, the research focuses on the linguistic patterns and regulatory obligations contained within privacy policies rather than the traits of the individuals those policies protect \[cite: 1, 10\]. This shifts the focus of the ethics protocol from human safety to data provenance and the protection of "corporate memory" \[cite: 3, 11\].

## **Ethical Mitigation of Interaction Risks**

By eliminating direct human interaction, the project mitigates several high-order ethical risks commonly associated with AI research, such as psychological impact and the "neurodivide" \[cite: 5\]. Research into brain-computer interfaces (BCIs) and neural-adaptive systems has highlighted that knowing thoughts or physiological states can be decoded can lead to a perceived loss of mental autonomy \[cite: 5\]. The PrERT-CNM framework avoids these tensions by operating as a "reflective" system rather than an "interpretive" one, analyzing the text of the law rather than the signals of the person \[cite: 5, 6\].

| Ethical Risk Category | Traditional Participant Model | PrERT-CNM Non-Participatory Model |
| :---- | :---- | :---- |
| **Informed Consent** | Requires written consent for each data use case \[cite: 6\]. | Relies on public document availability and research derogations \[cite: 7, 9\]. |
| **Data Sensitivity** | High (biometric, neural, health data) \[cite: 5\]. | Low (de-identified segment-level text) \[cite: 1, 4\]. |
| **Recruitment Burden** | High (IRB approval, volunteer management) \[cite: 6\]. | None (automated extraction from public repositories) \[cite: 1, 4\]. |
| **Traceability** | Limited by privacy-preserving anonymization \[cite: 12, 13\]. | High (end-to-end evidence paths to source records) \[cite: 1, 11\]. |

# **Academic Use Compliance and Corpus Licensing**

The governance of the project's data inputs is primarily defined by the licensing constraints of the datasets employed, most notably the Online Privacy Policies (OPP-115) corpus \[cite: 1, 4, 14\]. As the most widely utilized dataset in the privacy NLP domain, the OPP-115 provides a meticulously annotated baseline of 23,000 fine-grained data practices across 115 website policies \[cite: 4, 15, 16\].

## **Adherence to Research and Teaching Licenses**

Usage of the OPP-115 corpus strictly adheres to the standard research-and-teaching license constraints detailed in the project records \[cite: 1, 4\]. This license, formulated in the spirit of a Creative Commons Attribution-NonCommercial (CC BY-NC) agreement, explicitly limits the application of the data to scholarship and non-commercial academic research \[cite: 17\]. By embedding these constraints into the project’s data governance slide, the researcher ensures that the resulting PrERT-CNM model cannot be repurposed for commercial exploitation without violating the expert-annotated ground truth's original intent \[cite: 5, 17\].

The technical implications of this compliance are profound. The OPP-115 dataset provides segment-level annotations from ten distinct categories, such as "Third Party Sharing" and "Data Retention" \[cite: 14, 18\]. Maintaining strict license compliance allows the researcher to leverage these categories for "Phase 3" classification while preserving the academic integrity of the model's performance metrics \[cite: 1\].

## **Technical Resolutions for Taxonomy Coverage**

During development, "Taxonomy Coverage" was identified as a key technical challenge, specifically where auxiliary datasets like the APP-350 mapping reduced useful label coverage \[cite: 1\]. The ethical resolution to this technical limitation involved anchoring the model on the high-quality OPP-115 dataset while treating the APP-350 data as "negative evidence" or a secondary validation set \[cite: 1\]. This hierarchical approach to data quality ensures that the model’s reasoning remains grounded in the most rigorously annotated legal standards available, rather than being diluted by less granular mobile-app datasets \[cite: 15, 19\].

# **Pairing Public Corpora with Controlled Synthetic Records**

A significant innovation in the PrERT-CNM data governance strategy is the pairing of public research corpora with rigorously controlled synthetic records \[cite: 1, 4\]. This approach addresses the pervasive problem of "class imbalance," where standard datasets contain limited examples of rare or "stressed" system behaviors, which often caps overall macro-F1 gains in neural models \[cite: 1\].

## **The Evolution of Synthetic Data in Privacy Research**

Synthetic data (SD) is artificially generated information that maintains the statistical properties of real-world data without containing specific personal records \[cite: 12, 13\]. It has emerged as a vital solution for AI development in contexts where access to real data is restricted, scarce, or highly regulated \[cite: 20, 21\]. Within the PrERT-CNM pipeline, synthetic records are utilized in "Phase 2" to define metrics and establish baseline scenarios ranging from normal operation to adversarial conditions \[cite: 1\].

This methodology is supported by the emerging consensus that synthetic data will dominate real data in AI development by the end of the decade \[cite: 21\]. By generating artificial datasets that mimic the statistical distributions of real privacy policies, the project can expand its training reach to include edge cases—such as highly non-compliant or deceptive policy language—that are rarely found in the curated Alexa Top-115 list \[cite: 15, 20, 21\].

## **Ethical Pillars for Synthetic Governance**

The use of synthetic records is not a "neutral technical fix" but a socio-technical practice that requires layered governance \[cite: 20\]. The PrERT-CNM project applies several ethical pillars to ensure the integrity of its synthetic baseline:

1. **Fidelity and Utility:** The synthetic records must mirror the nuances of actual legal language to ensure the model's risk reasoning is transferable to real-world audits \[cite: 12, 20\].  
2. **Differential Privacy:** By mathematically limiting how much any single data point influences the synthetic output, the project prevents the risk of "overfitting," where the AI might accidentally memorize and regurgitate portions of the restricted source material \[cite: 13, 22\].  
3. **Cryptographic Provenance:** The system maintains secure, cryptographic logs of the parent dataset and the generation model used, keeping the synthetic records fully auditable \[cite: 1, 13\].

| Metric Type | Role of Public Corpora (OPP-115) | Role of Controlled Synthetic Records |
| :---- | :---- | :---- |
| **Ground Truth** | Provides expert-verified baseline for standard practices \[cite: 15, 17\]. | Simulates rare, adversarial, or "stressed" scenarios \[cite: 1\]. |
| **Scaling** | Limited to 115 policies; high-fidelity but small sample \[cite: 19\]. | Theoretically infinite scaling for model stress-testing \[cite: 21\]. |
| **Compliance** | Anchors the model in existing regulatory implementations \[cite: 14\]. | Proactively probes the model's ability to recognize novel compliance risks \[cite: 1, 23\]. |
| **Risk Reasoning** | Used to calibrate the Bayesian posterior on historical data \[cite: 1\]. | Used to define the bounds of the "stressed" metric landscape \[cite: 1\]. |

# **The Primacy of Professional Judgment Over Probabilistic Outputs**

Slide 18 specifies that system outputs act as decision support and must never replace expert professional or legal judgment \[cite: 1, 4\]. This claim aligns with the ethical guidance for generative AI issued by the State Bar of California, which emphasizes that any use of AI must not diminish or abdicate professional responsibility \[cite: 24\].

## **The Limits of Algorithmic Inference**

AI models, including the transformer architectures used in PrERT-CNM, produce probabilistic rather than deterministic outputs \[cite: 24\]. While the project achieved a primary model accuracy of 95.4% (PrivBERT Freeze), the existence of even a small error rate (such as the Brier Score of 0.04) necessitates a human-mediated implementation \[cite: 1\]. In the field of law, reliance on fabricated, incomplete, or biased outputs is inconsistent with professional obligations \[cite: 24\].

The PrERT-CNM project acknowledges these limitations through its architectural design. Rather than delivering a binary "pass/fail" compliance verdict, the system leverages a Bayesian posterior risk layer that produces scores (e.g., 0.901) paired with level-specific credible intervals and evidence traces \[cite: 1\]. This "uncertainty disclosure" allows the professional user to gauge the reliability of the system’s inference before implementation \[cite: 1, 23\].

## **Transformative Defaults and Value Trade-offs**

A critical second-order insight is the risk that AI systems can transform "episodic and revisable value judgments into durable, scalable defaults" \[cite: 25\]. If an AI system defaults to a high-throughput screening threshold that prioritizes speed over clinical or legal nuance, it redistributes risk and responsibility without active human intent \[cite: 25\]. By framing its outputs as "decision support" only, PrERT-CNM ensures that these value trade-offs remain contestable \[cite: 26\]. Professionals are encouraged to weigh the raw data and situational nuance against the system's "black box" resolution, preserving the embodied nature of human understanding \[cite: 25, 27\].

# **Core Principle: Transparent Evidence Links and Human-in-the-Loop**

The final and most critical pillar of the PrERT-CNM governance framework is the guarantee of safety through transparent evidence paths and active human oversight \[cite: 1, 4\]. Safety in high-stakes domains is viewed not as a technical property of the code, but as a socio-technical property of the collaborative human-AI system \[cite: 23, 27\].

## **Explainable AI (XAI) and Active Mediation**

The PrERT-CNM-v4 pipeline delivers regulation-aligned compliance assessments that include clause citations for every verdict \[cite: 1\]. This "Transparent Evidence Path" allows for auditability, enabling a human auditor to verify why a specific policy segment was flagged as a risk \[cite: 1\]. This approach moves beyond passive reporting, transforming Explainable AI (XAI) into an active mediation layer for human-AI collaboration \[cite: 23\].

The human-in-the-loop (HITL) paradigm specifically addresses the "brittleness" of AI models when faced with contextually complex threats \[cite: 23, 27\]. While the machine is adept at processing vast volumes of data to identify patterns, it lacks the ability to reason about situations outside its training distribution \[cite: 27, 28\]. The human role in the PrERT-CNM loop is to act as a "crucial filter" against factual errors and fabricated evidence, a process modeled on the appellate processes in judiciary decision-making \[cite: 28, 29\].

## **Cognitive Load and Human Factors**

Effective HITL design must balance the benefits of automation against the "cognitive costs" of human interaction \[cite: 27\]. Poorly designed automation can induce complacency or overwhelm operators with alerts \[cite: 27\]. The PrERT-CNM project addresses this by:

* **Structured Evidence Traces:** Presenting information in a cognitively friendly format that supports "bounded rationality" \[cite: 27\].  
* **Automatic Flagging:** Triggering human review only when the system's confidence falls below a specific threshold or for high-stakes compliance controls \[cite: 28\].  
* **Source Validation:** Requiring domain specialists to check the existence, accessibility, and accuracy of AI-generated citations to prevent hallucinations \[cite: 24, 28\].

| Framework Component | Technical Mechanism | Human Oversight Integration \[cite: 1\] |
| :---- | :---- | :---- |
| **Phase 1: Extraction** | Regulation-specific extraction of GDPR/ISO controls. | Expert validation of control relevancy. |
| **Phase 2: Metric Design** | Synthetic baseline scenario generation. | Definition of "stressed" vs. "normal" priors. |
| **Phase 3: Risk Layer** | Bayesian posterior risk calculation. | Interpretation of credible intervals and traces. |
| **Phase 4: Validation** | Calibration checks (ECE/Brier) and Gradio Demo. | Active implementación of audit verdicts. |

# **Information Life Cycle Management (ILCM) and Project Records**

Data governance within the PrERT-CNM project is further reinforced by strict adherence to Information Life Cycle Management (ILCM) principles as defined in ISO/IEC 15944-12 \[cite: 11, 14\]. This international standard mandates that organizations maintain systematic, IT-enabled record retention and disposal schedules (RRDS) that apply to both primary and secondary data collections \[cite: 11\].

## **Maintenance of Project Memory Records**

The project utilizes "Project Memory Records" to document key architectural decisions and reproducible evidence bundles \[cite: 1, 30\]. This ensures that the state changes of the research data—from raw standard extraction to the final validated model—are fully traceable and authentic \[cite: 11, 31\]. In the context of "Corporate Memory," these records preserve the decision-making logic that is often lost in final research outputs \[cite: 11, 32\].

ISO/IEC 27002:2022 provides additional guidance on protecting the reliability and usability of such records \[cite: 31\]. The PrERT-CNM pipeline implements these controls through:

* **Artifact-Driven Reproducibility:** Use of versioned JSONL files and manifests to ensure that each pipeline pass can be audited by third parties \[cite: 1\].  
* **Commit History Tracking:** Continuous and measured iteration on benchmarking protocols, code artifacts, and documentation quality in the project repository \[cite: 1\].  
* **Metadata Integration:** Each record contains structured metadata—describing context, content, and structure—which is an essential component for demonstrating compliance with privacy protection requirements (PPR) \[cite: 3, 11, 31\].

## **Disposition and Expungement Rules**

A key aspect of governance is the disposition of data at the end of the research lifecycle. Rule 042 of ISO/IEC 15944-12 stipulates that any personal information no longer relevant to operations must be disposed of immediately via "expungement"—the complete wiping or destruction of the record \[cite: 11\]. While the PrERT-CNM project focuses on non-personal legal texts, the governance of its synthetic records and expert annotations follows similar protocols to prevent commercial misuse or unauthorized disclosure \[cite: 5, 11\].

# **Convergence of Ethical Values and Technical Validation**

The PrERT-CNM project serves as a demonstration case for "Precedent-Based Professional Role Ethics" \[cite: 33\]. By combining LLMs with role-based ontologies, the system supports structured ethical reasoning that reflects established professional standards rather than attempting to replace them \[cite: 33\]. The use of a "self-evaluating pipeline" and "structured prompt profiles" ensures that the transition from initial inspiration to formal audit is bridged by rigorous measurement and peer-reviewed architectures \[cite: 32, 34\].

## **Calibration as an Ethical Requirement**

In the preliminary results of the PrERT-CNM pipeline, the primary model (PrivBERT Freeze) demonstrated not only high accuracy (95.4%) but also a low Expected Calibration Error (ECE) of 0.052 \[cite: 1\]. High accuracy in a model is a necessary but insufficient condition for ethical deployment; calibration is required to ensure that the system's "subjective" probability of a risk aligns with the "objective" frequency of that risk occurring \[cite: 14, 35\]. A poorly calibrated model could project high confidence in a false compliance verdict, leading to professional liability and regulatory non-compliance \[cite: 1, 24, 25\].

## **Protecting Divergent Creative and Analytical Trajectories**

To maintain the integrity of the research journey, the project employs multi-agent systems and "Blind Peer Review" architectures \[cite: 32, 34\]. These systems preserve divergent trajectories rather than defaulting to the content homogenization typical of standard conversational models \[cite: 34\]. This is essential for capturing the "tacit knowledge" of the legal annotators and ensuring that the final exegesis acts as a theoretical anchor for the major risk-quantification project \[cite: 32, 34\].

# **Future Roadmap for Governance and Safety**

The future plan for the PrERT-CNM project, leading to dissertation completion, involves moving from foundation-building to the extension of capabilities \[cite: 1\]. A key planned feature is the addition of "automated contradiction scoring" for policy-versus-practice analysis \[cite: 1\]. This will require even more stringent data governance, as the model will need to retrieve and compare external evidence—such as data breach reports or technical logs—against the text of the privacy policy \[cite: 1\].

As the system moves toward this stage, the "Core Principle" of human oversight becomes even more vital. The evolution toward agentic AI introduces the capacity to conclude tasks without contemporaneous review \[cite: 24\]. However, the PrERT-CNM framework maintains that this autonomy does not satisfy the professional's ethical obligations \[cite: 24\]. The "Safe implementation" of the future roadmap requires:

* **Periodic Reassessment:** Continuous evaluation of the system’s capabilities and risks as models evolve through updates \[cite: 24\].  
* **Audit-Ready Packaging:** Maintaining the Gradio and Hugging Face demo workflows in an "artifact-driven" manner to ensure all scaling milestones are converged and validated \[cite: 1\].  
* **Defensible Discussion:** Formulating a discussion of limits and impacts that explicitly articulates where the machine's labor ends and the human's judgment begins \[cite: 1\].

# **Conclusions: A Reproducible Evidence Bundle for Privacy AI**

The Ethics and Data Governance framework established in Slide 18 of the PrERT-CNM project represents a comprehensive response to the ethical challenges of automated privacy quantification \[cite: 1\]. By successfully integrating non-participatory collection with academic corpus licensing and rigorously controlled synthetic baselines, the project achieves a high degree of technical utility while maintaining legal compliance \[cite: 1, 11, 20\].

The project's refusal to replace expert judgment with probabilistic outputs—coupled with its guarantee of transparent evidence paths—sets a standard for postgraduate research in high-stakes AI domains \[cite: 1, 24, 27\]. Ultimately, this research positions the transformer model as a "named partner" in a disclosed collaboration, facilitating a more sustainable and ethical relationship between human cognition and computational pattern recognition \[cite: 6, 32\]. The quantification of user privacy risk is thus not just a mathematical task but a socio-technical one, preserved within a reproducible evidence bundle that is audit-ready and aligned with international excellence \[cite: 1, 11\].

# **Sources**

1. [Viva](https://drive.google.com/open?id=14cnx2xpW-oErKU2ISmZuKnJ8nYzImjw7zqJB4oVLJYc)  
2. [main.bbl](https://drive.google.com/open?id=1yIUmIANwIbIH41qyseRL4E39jLAf5PFe)  
3. [NIST-1.1.docx](https://drive.google.com/open?id=1GLbly83eO3b0TeFhDmkHb0OlW8GkXQ9f)  
4. [Introducing the PrivaSeer Corpus of Web Privacy Policies \- ACL Anthology](https://aclanthology.org/2021.acl-long.532.pdf)  
5. [PhD Brain AI](https://drive.google.com/open?id=1O-qxBiSvAvuijaL0NAJBoTvAGOp71vc5eQKMazHMULE)  
6. [Masters Proposals](https://drive.google.com/open?id=1-LzRoAvdhxR-E7gdZGwJYarti2rL6_cu4s9Ncdy3x6o)  
7. [GDPR-2016\_679.docx](https://drive.google.com/open?id=15oSwrlUkMQeFfVy354WY0Iwn9s_7GbDi)  
8. [BS EN ISO-IEC 29100-2020.docx](https://drive.google.com/open?id=13SP1dv63n_-pl8nc5r9EHfAAhNCnkou7)  
9. [GDPR-2016\_679.pdf](https://drive.google.com/open?id=11-5AJEYYMLgbBJcUSlPxSpZ1Rt-1ZjER)  
10. [Characterisation and Quantification of User Privacy: Key Challenges, Regulations, and Future Directions | Request PDF \- ResearchGate](https://www.researchgate.net/publication/387182924_Characterisation_and_Quantification_of_User_Privacy_Key_Challenges_Regulations_and_Future_Directions)  
11. [BS ISO-IEC 15944-12-2025.docx](https://drive.google.com/open?id=1VVPMJ518C3kMKhAk1pKsY_aWwAkVtUbK)  
12. [Synthetic data in medical imaging within the EHDS: a path forward for ethics, regulation, and standards \- Frontiers](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1620270/full)  
13. [What ethical and legal framework is required for AI-generated synthetic data that is based on locally sovereign data, when the resulting synthetic data is transferred globally? \- Quora](https://www.quora.com/What-ethical-and-legal-framework-is-required-for-AI-generated-synthetic-data-that-is-based-on-locally-sovereign-data-when-the-resulting-synthetic-data-is-transferred-globally)  
14. [references.bib](https://drive.google.com/open?id=1qA0QBqY0vLBPQDoMGSw_cV_pRMGcclR7)  
15. [Privacy Policies of IoT Devices: Collection and Analysis \- PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8914639/)  
16. [Large Language Models: A New Approach for Privacy Policy Analysis at Scale \- arXiv](https://arxiv.org/html/2405.20900v1)  
17. [OPP-115 Corpus (ACL 2016\) \- Usable Privacy Policy Project](https://usableprivacy.org/data)  
18. [arXiv:2212.10011v2 \[cs.CL\] 12 May 2023](https://arxiv.org/pdf/2212.10011)  
19. [A Large Publicly Available Corpus of Website Privacy Policies Based on DMOZ](https://identity.utexas.edu/sites/default/files/2020-12/A%20Large%20Publicly%20Available%20Corpus%20ofWebsite%20Privacy%20Policies.pdf)  
20. [Ethics of Synthetic Data: A Lifecycle Perspective \- UCL](https://www.ucl.ac.uk/engineering/sites/engineering/files/2025-11/05%20-%20Ethics%20of%20Synthetic%20Data.pdf)  
21. [Changes in the Privacy Landscape in Recent Years: Analysis From AI and Non-AI Contexts](https://www.computer.org/csdl/magazine/co/2025/09/11134648/29olYuIYfcY)  
22. [A Sensitivity-Aware and PSO-Driven Differential Privacy Method With Customized Budgets for Structured Data Perturbation \- IEEE Computer Society](https://www.computer.org/csdl/journal/tk/2026/05/11430682/2eNCct1YG7m)  
23. [Human-in-the-Loop Explainable AI for Reliable Autonomous Cybersecurity Infrastructure](https://www.preprints.org/manuscript/202601.2031)  
24. [THE STATE BAR OF CALIFORNIA STANDING COMMITTEE ON PROFESSIONAL RESPONSIBILITY AND CONDUCT PRACTICAL GUIDANCE FOR THE USE OF GENE](https://www.calbar.ca.gov/Portals/0/documents/ethics/Generative-AI-Practical-Guidance.pdf)  
25. [Protecting clinical value judgment in the age of AI \- PMC \- NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC13035987/)  
26. [AUTOMATED DECISION SUPPORT TECHNOLOGIES AND THE LEGAL PROFESSION](https://btlj.org/data/articles2019/34_3/04_Kluttz_Web.pdf)  
27. [Human-in-the-Loop Artificial Intelligence: A Systematic Review of Concepts, Methods, and Applications \- MDPI](https://www.mdpi.com/1099-4300/28/4/377)  
28. [What is Human in the Loop: Verifying AI Citation Trust \- Medium](https://medium.com/@barrettrestore/what-is-human-in-the-loop-verifying-ai-citation-trust-2f51a41647ec)  
29. [How AI can learn from the law: putting humans in the loop only on appeal \- PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10457290/)  
30. [How to Build License-Compliant Synthetic Data Pipelines for AI Model Distillation](https://developer.nvidia.com/blog/how-to-build-license-compliant-synthetic-data-pipelines-for-ai-model-distillation/)  
31. [BS EN ISO-IEC 27002-2022.docx](https://drive.google.com/open?id=1KsFX7fK6fOMCnDUc6rrR2tJ8lcB9zX8D)  
32. [Training Pipelines and Open-Source Model Selection](https://drive.google.com/open?id=1cL-y_Y8ewSKae2WCZVh44OgvhmClhn6saTthfvyD60A)  
33. [Precedent-Based Professional Role Ethics for AI Decision Analysis](https://ojs.aaai.org/index.php/AIES/article/view/36794)  
34. [The Architecture of Algorithmic Creativity: Pipelines, Datasets, and Open-Source Frameworks in Postgraduate Research](https://drive.google.com/open?id=1BpT3sKuiIkS73T_jv5cd0KnEPll2a872ezLA4SClBfo)  
35. [References](https://drive.google.com/open?id=1uBSARyhUvyUgDTdoq8dytTPRVR9GxS8mAuF2bbLSWnA)

