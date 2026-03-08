I anchored the plan around **Triton fused softmax** because Triton’s official tutorial presents it as a practical fused, bandwidth-bound kernel that can be significantly faster than PyTorch for rows that fit in on-chip memory, while Triton’s `do_bench` API gives you a documented way to benchmark median runtime over repeated runs. That makes it a strong hackathon demo target without the complexity of full GEMM autotuning. ([Triton Language][2])

The spec makes one deliberate change from your original surrogate-discovery reward: it uses **dense step reward** equal to `prev_validation_mse - new_validation_mse`. That preserves the same total objective over an episode, but it is easier to debug and more convenient if you later plug the environment into an RL trainer.

The most important implementation choices in the spec are:

1. **Measure first, then environment.**
   The coding agent must build `scripts/collect_measurements.py` first and generate `data/softmax_measurements.csv` on the H100. The OpenEnv environment then uses those real measurements as its oracle rather than timing kernels inside every rollout.

2. **Each episode is one shape.**
   A reset picks a task like `softmax_m4096_n2048`, seeds the surrogate with 5 measured configs, and the agent spends a fixed budget proposing new normalized 3-vectors that map to discrete `(BLOCK_SIZE, num_warps, num_stages)` configs.

3. **Use OpenEnv natively.**
   The spec uses the standard scaffold and API shape OpenEnv documents: `openenv init`, `models.py`, `client.py`, `openenv.yaml`, `server/app.py`, and an environment class exposing `reset`, `step`, and `state`. ([GitHub][1])

4. **Prefer local Python process first, Docker second.**
   OpenEnv’s docs show local development through editable install plus `uv run server`, and TRL’s OpenEnv docs explicitly note you can also run environments locally as a Python process or via Docker. For H100 development, the spec makes local Python the default path and Docker an acceptance-stage packaging step. ([GitHub][1])

5. **Don’t hardcode a stale PyTorch wheel command.**
   The spec tells the coding agent to install PyTorch using the current official “Get Started” selector for Linux + pip + the correct CUDA platform, then verify `torch.cuda.is_available()`, instead of assuming a fixed wheel URL. ([PyTorch][3])

If you want a shorter paste-in brief for the coding agent, use this:

```text
Build a local OpenEnv environment named softmax_surrogate_env on this H100 machine.

Goal:
- Demonstrate surrogate-guided autotuning of a Triton row-wise softmax kernel using real H100 measurements.
- Do not use Prime.
- Use OpenEnv directly.

Requirements:
- Scaffold with `openenv init softmax_surrogate_env`.
- Implement OpenEnv-native files: models.py, client.py, openenv.yaml, server/app.py, server/softmax_surrogate_environment.py.
- Environment must expose reset(), step(), and state().
- Action is JSON: {"x": [float, float, float]} with values clamped to [-1, 1].
- Internal mapping: normalized x -> nearest discrete config from a measured catalog.
- Kernel family: Triton row-wise softmax, fp16, shapes M=4096 and N in {256,512,1024,1536,2048,3072,4096,6144,8192}.
- Tunable axes: BLOCK_SIZE in {256,512,1024,2048,4096,8192}, num_warps in {1,2,4,8}, num_stages in {1,2,3,4}. Skip invalid BLOCK_SIZE < N.
- First implement scripts/collect_measurements.py and generate data/softmax_measurements.csv on the H100.
- Validate every config against torch.softmax.
- Benchmark with median latency over repeated runs.
- Store median_ms, effective_gbps, and score=-log(median_ms).
- reset(): choose a task, seed surrogate with 5 measured configs, compute validation MSE over all measured configs for that task.
- step(): add one measured config, refit surrogate, return observation including latency, chosen config, validation_mse, steps_remaining.
- reward per step = previous_validation_mse - new_validation_mse.
- done when budget is exhausted.
- Implement scripts/smoke_test_client.py, run_random_baseline.py, run_surrogate_baseline.py, demo_compare.py.
- demo_compare.py must show fixed heuristic vs random vs surrogate baseline on one fixed task and then rerun the winning config live on the H100.
- Write all logs and evals under outputs/.
- Finish only when local server works, smoke test passes, and surrogate beats random on at least one fixed task.

Use the detailed spec file as the source of truth.
```

The full spec already includes the repo tree, exact deliverables, acceptance criteria, failure handling, and command sequence. If helpful, I also have the starter measurement collector and earlier surrogate env draft here: [collector](sandbox:/mnt/data/collect_softmax_measurements.py) and [earlier env draft](sandbox:/mnt/data/kernel_softmax_surrogate_env.py).

[1]: https://github.com/meta-pytorch/OpenEnv/blob/main/README.md "OpenEnv/README.md at main · meta-pytorch/OpenEnv · GitHub"
[2]: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html?utm_source=chatgpt.com "Fused Softmax"
[3]: https://pytorch.org/get-started/locally/?utm_source=chatgpt.com "Get Started"

