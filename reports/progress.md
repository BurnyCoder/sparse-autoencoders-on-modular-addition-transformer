# Progress

Author: Libor "Burny" Burian.

## 2026-11-27, stage 1 (before any results)

**Done**
- Environment: uv project on Python 3.13, PyTorch 2.11 + CUDA 12.8 (checked on the RTX 5070, sm_120), SAELens 6.56.4, TransformerLens 4.4.1. The public GitHub repo is created.
- Literature review: [literature-review.md](literature-review.md) and [state-of-the-art-review.md](state-of-the-art-review.md). An independent fact-check is running (results go to [reviews/](reviews/)).
- Stage-1 preregistration: [preregistration.md](preregistration.md). Run matrix: [planned-runs.md](planned-runs.md). Plan: [plan.md](plan.md).

**Now**: pushing the preregistration before writing any training code or training any model.

**Next**:
- TDD implementation of the pipeline, then the dev base model (seed 999) and the dev SAEs and steering exploration.
- Stage-2 freeze, then the confirmatory seeds 0–9, controls, ablations, paper, site and HF collection.

**Overall**: about 10% of the plan done (setup, literature review, preregistration). Compute is small (under about 4 GPU-hours in total). Most of the remaining time goes to implementation, analysis and writing.

**Results so far**: none. No model has been trained yet.
