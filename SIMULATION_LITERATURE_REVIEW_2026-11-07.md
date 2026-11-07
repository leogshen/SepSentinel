# Simulation for precursor discovery and sepsis early warning

Focused literature review,7 November2026. This is a targeted narrative review, not a systematic review or proof of novelty. Primary papers were searched for sepsis/inflammation mechanistic simulation, clinical time-series generation, digital twins and hybrid physiological learning. Older foundational work and close2025–2026 preprints are included; publication status and demonstrated task matter more than venue prestige. No new training was run for this review.

## Scientific purpose, confirmed by the user

Develop a physiologically meaningful generator that helps discover coordinated precursors relative to each patient's baseline, then use those trajectories and hypotheses to teach an early-warning model. Real-patient early-warning utility is the eventual test. Simulation is both a scientific instrument and a source of training experience.

## Main finding

There is substantial prior work on both halves of this ambition. We should not claim that mechanistic synthetic sepsis trajectories or simulation-assisted prediction are new. The potentially valuable contribution is a demonstrable chain from mechanism, through robust observable precursors, to improved real-patient warnings. No single reviewed paper supplies a turnkey clinically validated generator for all of our inputs and this complete claim.

### 1. Mechanistic inflammation and physiological simulation

**Reynolds et al.,2006, Journal of Theoretical Biology.** Four differential equations connect pathogen, activated phagocytes, tissue damage and anti-inflammatory activity. Feedback supports different disease outcomes and exploration of interventions. This is a compact conceptual foundation for progression, not a calibrated34-variable EHR generator. [Paper](https://doi.org/10.1016/j.jtbi.2006.02.016).

**Cockrell and An,2017, Journal of Theoretical Biology.** The IIRABM line represents cellular interactions and stochastic inflammatory dynamics, exploring different outcome regions rather than one deterministic course. It is directly relevant to recovery/deterioration boundaries and uncertainty around a proposed cliff. Its simulated outcome space is not evidence that every human event has an observable precursor. [Paper](https://pubmed.ncbi.nlm.nih.gov/28728997/).

**Larie, An and Cockrell,2021, Frontiers in Physiology.** Neural networks forecast trajectories produced by the immune-response agent-based model. Sequential updates are needed as forecasts extend into uncertain futures. This directly precedes the simulator-to-learning part of our idea, but forecasts of the simulator do not demonstrate transfer to clinical patients. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8552109/).

**Clinical calibration/rule discovery,2021.** The An–Cockrell group used genetic algorithms to refine a high-dimensional inflammatory model using heterogeneous clinical observations. This is relevant to fitting a distribution of plausible mechanisms rather than a single average patient. The article links `An-Cockrell/IIRABM_MRM_GA`, but that URL returned404 during this review: code access is unresolved, not verified available. [Paper](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2021.662845/full).

**Mathematical modeling of septic shock based on clinical data,2019.** Couples immune, cardiovascular, nervous-system and pharmacological components. It fits heart-rate/blood-pressure treatment courses for three patients. Useful as a bridge from latent inflammation to observable physiology; three retrospectively adjusted cases do not establish population-level early prediction. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6404291/).

**Lal et al.,2020, Critical Care Explorations.** Hybrid agent-based/discrete-event/Bayesian modeling represents organ systems and treatment response during the first24h of sepsis. It addresses treatment-response simulation after sepsis, not our central presymptomatic detection task. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7671877/).

### 2. Learned synthetic clinical trajectories

**Esteban, Hyland and Rätsch,2017, RCGAN preprint.** An early recurrent conditional GAN approach for real-valued medical sequences. This is a methodological precursor for statistical trajectory generation; it does not by itself provide physiological explanations. [Paper](https://arxiv.org/abs/1706.02633).

**Li et al.,2023, EHR-M-GAN, npj Digital Medicine.** Joint generation of continuous and discrete longitudinal ICU data, evaluated on MIMIC-III/eICU/HiRID. It also evaluates synthetic augmentation for medical-intervention prediction. This is a strong practical baseline for joint trajectories and treatment variables, but that endpoint differs from precursor discovery before sepsis. [Paper](https://www.nature.com/articles/s41746-023-00834-7). An author-linked TensorFlow implementation is available; old dependencies require environment work, and it was not installed here. [Code](https://github.com/jli0117/ehrMGAN).

**Yoon et al.,2023, EHR-Safe, npj Digital Medicine.** Generates mixed longitudinal records including missingness and evaluates statistical fidelity, downstream utility and privacy attacks on MIMIC-III/eICU. Useful for modeling observation patterns. Preserving dataset utility is a different claim from discovering mechanisms or improving sepsis warning. No author-verified runnable implementation was established in this review; do not confuse similarly named third-party repositories with the original. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10421926/).

### 3. Direct contemporary overlap

**Chang et al.,2025, Organ-Agents (arXiv preprint).** Organ-specific LLM agents generate interacting physiological trajectories. The paper reports held-out/external simulation checks and clinician review. Crucially, its abstract describes synthetic-trained early-warning classifiers, whereas Section2.7 explicitly describes real-trained classifiers tested on generated/counterfactual data. Until resolved, this supports an author-reported fidelity experiment, not established synthetic-to-real training benefit. Weijie Wang and Lanjun Wang are coauthors; this might be the user's earlier “Wang” reference, but that identity is not confirmed. [Full text, especially2.7](https://arxiv.org/html/2508.14357v1).

**Nie et al.,2026, LLM-guided temporal simulation for early warning (arXiv preprint).** Forecasts physiological indicators and combines them with observed history for classification. Direct conceptual overlap. Its stated onset alignment, first-SOFA>=2 marker, imputation and clinical-summary timing require careful auditing before interpreting the reported AUCs as causal online sepsis prediction. A first absolute SOFA threshold alone is not a complete infection-related sepsis definition. The paper mentions ICD selection, but its labeling details need reconstruction. Weijie Wang is a coauthor here too; no inference that this is the intended reference. [Full text](https://arxiv.org/html/2604.20924v1).

**Backes et al.,2026, Frontiers in Network Physiology.** A Latent Dynamics Model embeds a coupled-oscillator organ/immune network into learning from MIMIC-IV. This directly overlaps with interpretable physiological states and trajectory learning. It is primarily a constrained predictive representation, not demonstration that independently generated precursor examples improve external early warning. The paper promises a public repository; its runnable status was not established here. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13327882/).

## Distinctions to preserve when reading results

1. Reconstructing a trajectory after seeing it is not forecasting that trajectory from its prefix.
2. Training on real records and testing on synthetic records asks whether generation preserves familiar patterns. Training on synthetic records and testing on untouched real records asks a different transfer question.
3. Synthetic-plus-real improving over real-only is stronger evidence of augmentation utility, but needs equal optimization, resampling, and simple-augmentation controls.
4. Physician plausibility ratings, marginal distributions and low forecast error do not establish a correct causal intervention model.
5. A trend specified by the simulator is not a discovered human biomarker. It is a hypothesis until independently corroborated.
6. “Digital twin” should not imply individualized validated physiology or ongoing patient synchronization unless those capabilities were actually demonstrated.

## Implications for our design (our synthesis, not established results)

A small hybrid model is a defensible next step. Begin with published infection/inflammation/damage/recovery dynamics. Add an observation model mapping those states to a limited set of measured channels, then a separate acquisition model controlling sampling, measurement noise and release delays. Estimate uncertain observation relationships from real training data, preserving uncertainty rather than imposing a single hand-written mapping as truth.

Do not immediately simulate every CinC variable. First choose channels whose relationships can be defended and whose repeated measurements exist. Cytokine dynamics and their translation to routine labs are a central uncertainty; naming a latent variable “inflammation” does not make it measured IL-6. PSU biomarker coverage is still unverified.

Represent an ensemble of plausible mechanisms, baselines and treatment responses. Search for precursor features that persist across several plausible models, rather than features special to one equation set. Include infectious recovery, noninfectious injury, chronic abnormalities, interventions, and cases with no recoverable warning. A cliff can be gradual, abrupt, stochastic or observation-induced; do not assume a universal tipping-point signature.

Candidate hypotheses might involve cross-signal timing, change relative to an individual's baseline, or loss of recovery after a disturbance. These are candidates for testing, not established sepsis biomarkers. Evaluating a precursor requires showing when it becomes detectable using only available observations, whether controls show it too, and whether it adds information beyond current state and measurement behavior.

## Recommended sequence

1. Reproduce qualitative regimes of a published low-dimensional inflammatory model and document its assumptions/units/parameter sources. This replaces arbitrary core dynamics in our current diagnostic pilot.
2. Test structural uncertainty using more than one defensible mechanistic formulation or alternative feedback mechanisms. Use agents if cellular heterogeneity is necessary for the hypothesis; do not adopt a large agent system only for complexity.
3. Fit/calibrate the measurement and sampling layers on real training data. New PSU data are not yet available; existing development data can support preparation without being described as fresh validation.
4. Predefine precursor hypotheses and train current-state, temporal-summary and sequence controls. Compare generic synthetic augmentation with mechanism-targeted augmentation at equal compute.
5. Test on real held-out patients with frozen timing and alarm accounting. Preserve a new PSU subset for confirmation if identifiers, temporal resolution, endpoint and approval permit.

Possible contribution: a reproducible mechanism-to-measurement framework that identifies robust precursor hypotheses and demonstrates real early-warning benefit under sparse observation. This is a research target, not a novelty claim already established by this review.

## Relation to work already completed

Our `synthetic_trajectory_pilot.py` uses illustrative latent dynamics and a manually specified channel mapping. It demonstrated controlled temporal learnability and simulated augmentation benefit, not biological validity. Its high simulated prevalence, source-family test mechanisms and single seed prevent clinical interpretation. Keep it as an engineering test; do not promote it into a physiological discovery model by relabeling it.

The official COMPOSER evaluator audit is useful for keeping evaluation honest but is separate from the simulation scientific contribution. PSU TriNetX planning remains documented in `PSU_TRINETX_PLAN.md`. The user now explicitly confirms both aims: precursor discovery and training an early-warning model.

## Review limits

This search did not independently reproduce any paper or validate its clinical claims. Some code availability remains unresolved. The recent arXiv items are treated as preprints; no prestigious journal acceptance is inferred from template text or author assertions. Exhaustive diffusion-generator coverage, all digital-twin software engines, and a systematic citation-network search remain outside this focused review. The most direct modern papers require detailed cohort/label/leakage audits before borrowing their reported performance targets.
