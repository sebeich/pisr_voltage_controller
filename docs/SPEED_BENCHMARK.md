# Computational-speed benchmark (Sec. V-C)

This document describes how the PISR speed-up is measured. It covers the baseline solver, warm-start, timing method, hardware and software versions.

Reproduce with:

```bash
experiments/V-C_offline_control_fig9/run.sh bench   # ~3 min
```

The results, including the environment they were measured on, are written to `results/V-C_offline_control_fig9/benchmark_{pisr,powerflow,summary}.json`.

## What is compared

The optimizer is the same in both cases: cost function (1), differential evolution (BlackBoxOptim `de_rand_1_bin_radiuslimited`, population 20), the same bounds and the same stopping rule. The two cases differ only in the inner voltage model that evaluates \|V\|max for a candidate setpoint.

| | PISR surrogate | Baseline |
|---|---|---|
| Model | closed-form PISR equations (`models/offline_highvar`, 5 target buses) | full AC power flow of the same grid |
| Software | Julia, `eval_tree_array` of SymbolicRegression.jl | pandapower `runpp` (Newton–Raphson, numba-compiled) |
| Output per evaluation | max \|V\| over the 5 prosumer buses | all bus voltages; max over the same 5 buses |

The speed-up is the ratio of the time per inner evaluation, because the number of evaluations per optimization is the same for both models. The baseline is an AC **power flow** inside the identical optimizer. The interior-point AC-OPF (`pandapower.runopp`) is not used for the speed-up.

## Baseline solver settings

| Setting | Value |
|---|---|
| Solver | pandapower 3.4.0 `runpp`, `algorithm="nr"` (Newton–Raphson) |
| Initialisation / warm start | `init="flat"` on **every** solve. No warm start from the previous candidate or time step (no `init="results"`), and no reuse of admittance matrices between solves |
| Convergence | `tolerance_mva = 1e-7`, `max_iteration = 30` |
| Options | `calculate_voltage_angles=True`, `enforce_q_lims=True`, `numba=True` (JIT-compiled Newton–Raphson, compiled before timing) |
| Network | CoSES LV grid (Sec. IV-A), built by `src/services/powerflow_solver_api.py` |
| Timed region | the `pp.runpp(...)` call only, including pandapower's internal model build and result write. No HTTP and no network copy |

## Timing method

- **Inputs:** the same 100 test inputs for both models (`data/offline/test_data_complex_blockrand.csv`), evaluated one input vector at a time, as inside the optimizer.
- **Warm-up:** each input is evaluated once before timing, which removes JIT compilation.
- **Statistic:** the median per input, then the median over the 100 inputs.
  - PISR in Julia: BenchmarkTools `@benchmark`, 0.2 s per input.
  - `runpp`: 20 timed repetitions per input.
  - PISR in Python: 50 × 100 calls per input.
- **Language-matched check:** the PISR equations are exported from Julia and evaluated as plain Python complex arithmetic. This gives a Python-vs-Python ratio in addition to the deployed Julia path. The script asserts that both evaluations agree to 1e-9.
- **Execution:** single process, single thread, CPU governor set to `performance`.

## Hardware and software

| | |
|---|---|
| CPU | AMD Ryzen AI MAX+ PRO 395 (16 cores / 32 threads, up to 5.19 GHz, 64 MB L3) |
| Memory | 32 GB (31 GiB usable) |
| OS | Ubuntu 24.04.4 LTS, Linux 6.17 |
| Python stack | Python 3.12.3, pandapower 3.4.0, numba 0.65.1, numpy 2.3.5, scipy 1.16.3 |
| Julia stack | Julia 1.12.6, SymbolicRegression.jl 1.12.0, DynamicExpressions.jl 1.10.3, BlackBoxOptim.jl 0.6.3, BenchmarkTools.jl 1.6.3 |

## Results

Measured on 2026-10-02 (`results/V-C_offline_control_fig9/benchmark_summary.json`). All values are per evaluation, i.e. one candidate setpoint.

| Inner voltage model | Median time | p5–p95 | Speed-up vs. `runpp` |
|---|---|---|---|
| PISR, Julia `eval_tree_array` (controller path) | 2.22 µs | 2.19–2.25 µs | **2110×** |
| PISR, plain Python (same runtime as pandapower) | 7.6 µs | | **618×** |
| PISR, Julia `MLJ.predict` on a 1-row table (high-level API) | 34.3 µs | | 136× |
| pandapower `runpp` (baseline) | 4.67 ms | 4.64–4.72 ms | 1× |

- Accuracy of the PISR \|V\|max against the power flow on the same inputs: mean 6.4e-04 p.u., max 1.3e-03 p.u.
- The machine was otherwise idle, apart from an unrelated background service that used about 5 of the 32 hardware threads. The benchmark itself runs single-threaded. A run during a concurrent CIL experiment gave similar ratios (627× Python vs. Python, 2577× Julia).

## Notes

- The optimization runs `run.sh` (PISR) and `run.sh speed` (power flow) exercise the complete loop.
  - In `run.sh speed`, every power flow is a request to a local HTTP solver service. Its time per evaluation includes the request overhead, so it is not used for the speed-up.
  - The full optimization time also depends on the optimizer's wall-clock budget (`GLOB_MAXTIME`, see `docs/SETTINGS.md`).
- The "up to 500×" quoted in the paper refers to the earlier PHiL prediction study and is not re-measured here.
