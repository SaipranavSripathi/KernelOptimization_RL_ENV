# One Slide Outline

## Title

Measured GPU Autotuning on H200: Surrogate Search Beats Random and Produces Faster Triton Kernels Than `torch.compile`

## Slide Structure

### Left: What We Built

- Discrete-action GPU kernel autotuning benchmark
- Real measured oracle on H200
- Families:
  - `softmax`
  - `layernorm`
  - `grouped_gemm`
- Evaluation:
  - held-out shapes
  - held-out family

### Center: Search Results

- Held-out shapes:
  - `98.4%` lower regret AUC vs random
  - `+15.8 pts` oracle-hit rate
- Held-out family:
  - `66.9%` lower regret AUC vs random
  - `+16.7 pts` oracle-hit rate

Caption:

- Surrogate wins on search efficiency, especially under short tuning budgets.

### Right: Runtime Results

- Surrogate-selected Triton vs `torch.compile`
  - `softmax`: `3.32x` faster
  - `layernorm`: `2.86x` faster
  - `grouped_gemm`: `2.42x` faster

Caption:

- The kernels found by the tuner are not just good in theory; they are faster in live runtime.

## Bottom Strip

Key message:

We built a measured autotuning benchmark that shows both:

- better optimizer behavior than random
- better runtime than `torch.compile`

## Speaker Notes

- AUC means how quickly the optimizer gets close to the best kernel under a fixed search budget.
- Final endpoint differences vs random can be modest, but the surrogate reaches strong kernels much earlier.
- This is already useful as an autotuning platform for LLM-relevant kernels.
