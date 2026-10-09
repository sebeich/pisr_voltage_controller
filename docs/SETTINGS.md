# Settings used for the paper results

Every value below is the default in the repository code. The experiment drivers in `experiments/` use these values unless an environment variable overrides them. Power is in MW / Mvar unless stated otherwise.

## Digital twin (Sec. IV-A)

| Item | Value | Where |
|---|---|---|
| Grid | CoSES LV grid, 0.4 kV, up to 61 cable segments (standard types cfg16/50/70/95/150) | `data/grid/coses_matpower_source.xlsx`, `src/grid/coses_net.py`, `src/services/*` |
| MV equivalent | Thevenin source, 30 MVA short-circuit power, X/R = 3, 20 kV | same |
| Transformer | 250 kVA, 20/0.4 kV, Dyn, u_k = 4 %, u_kr = 0.5 %, P_fe = 0.5 kW, i_0 = 0.2 % | same |
| Angle reference | bus 33 (all phasors relative to it) | same |
| Scenarios | AR(1), α = 0.95, 3600 s at 1 s; high-variability μ_P = 5, σ_P = 3 (uncorrelated); correlated μ_P = 10, σ_P = 0.8, ρ = 0.85; P scaled ×5, Q ×0.5 (kW/kvar) | `data/scenarios/generate_scenarios.jl` |
| Prosumer buses (scenario columns 1..n) | offline: 64, 57, 55, 53, 61; real-time twin: 64, 57, 55, 61 | `coses_net.py`, `DESIRED_SCENARIO_BUS_ORDER` in the services |
| Controllable inverters | buses 55, 57, 64; bus 61 is uncontrolled and also carries the disturbance step | controllers, `CONTROLLED_BUSES` |

### Power flow (pandapower 3.4.0)

| Use | Settings |
|---|---|
| Offline datasets and Fig. 9 replay (`src/grid/coses_net.py`) | `runpp`, Newton–Raphson, flat start, `tolerance_mva = 1e-8`, `max_iteration = 50`, voltage angles on, Q limits enforced, numba |
| Real-time twins and speed baseline (`src/services/*.py`) | `runpp`, Newton–Raphson, flat start, `tolerance_mva = 1e-7`, `max_iteration = 30`, voltage angles on, Q limits enforced, numba. If that fails to converge: DC init, `tolerance_mva = 1e-6`, 50 iterations |
| Optional AC-OPF mode of the twins (`/solver` mode `opf`) | `runopp(calculate_voltage_angles=True, delta=1e-16)`, cost weights 30 (P) / 1 (Q), voltage band 0.97–1.05 p.u. in the routed twin (env `P_WEIGHT`, `Q_WEIGHT`, `VMIN`, `VMAX`); the long-run twin (`realtime_powerflow_service_with_noise.py`) hard-codes 0.90–1.05 p.u. Not used for any paper figure |

### Real-time loops (Sec. IV-B)

| Loop | Rate | Setting |
|---|---|---|
| Scenario replay | 1 Hz | `--scenario-rate 1` |
| Plant power flow | 20 Hz (50 ms) | `--pf-rate 20` (Fig. 8 data collection used 15 Hz) |
| Controllers | 10 Hz (100 ms) | `LOOP_PERIOD_S = 0.1` |
| Measurement noise (Fig. 10, phase 3) | σ = 5·10⁻³ p.u. on Re and Im of every published voltage, σ = 10⁻⁵ on published powers; plant state stays noise-free | `--v-noise`, `--s-noise`, `POST /noise` |
| Topology change | breaker K4 (Q1 in the paper) between buses 7 and 8, closing it forms a ring | `POST /breaker?closed=true` |

## PISR surrogate (Sec. III-B, Algorithm 1)

| Item | Value |
|---|---|
| Package | SymbolicRegression.jl 1.12.0 (`MultitargetSRRegressor` via MLJ 0.22) |
| Binary operators | `+ - * /`, `line_current(V, S) = conj(S / V)`, `voltage_drop(I, Z) = I·Z` (algebraically identical to `*` for complex arguments, so the effective operator set equals Algorithm 1 of the paper) |
| maxsize | 30 |
| npopulations | 3 × (CPU threads − 2) |
| Iterations | incremental: 20, then +100 per step until 1020 (`INITIAL_ITER = 20`, `STEP_ITER = 100`, `MAX_TOTAL_ITER = 1000/1020`) |
| Early stop | worst-bus max \|V\| error < `THRESHOLD = 0.001` p.u. (not reached in the paper runs) |
| Inputs / targets | complex powers S of the prosumer buses → complex voltages V of the same buses; slack voltage not used as input |
| Offline training rows (Fig. 7) | 100 rows drawn at random (`TRAIN_SEED = 0`) from the 500-sample set |
| API-collected sets (Fig. 8) | 50 training samples, 30 block-randomized test samples, perturbations U(15, 80) kW/kvar, seeds 4242 + 1000·i |

## Optimization (Sec. III-A, cost function (1))

The cost is J = w_P ΣΔP² + w_Q ΣΔQ² + λ·max(0, \|V\|max − V_limit)², with w_P = 30, w_Q = 1 and V_limit = 1.05 p.u. All controllers use BlackBoxOptim 0.6.3 with `de_rand_1_bin_radiuslimited` and population 20. BlackBoxOptim stops when its wall-clock budget `MaxTime` runs out, which happens before `MaxFuncEvals` is reached.

| Experiment | Script | λ | ΔQ bound | ΔP bound | MaxTime | Period |
|---|---|---|---|---|---|---|
| Fig. 9, PISR | `src/controllers/mpc_noslack.jl` | 1e4 | ±0.1 | [0, 0.1] | 0.1 s per sample | offline |
| Speed baseline | `src/controllers/mpc_powerflow.jl` | 1e4 | ±0.1 | [0, 0.1] | 20 s per sample | offline |
| Fig. 10, PISR | `src/controllers/rt_controller_showerror_routed.jl` | 1e10 | ±0.1 | ±0.3 | 0.05 s | 0.1 s |
| Fig. 10, sensitivity | `src/controllers/rt_sensitivity_traindata_routed.jl` | 1e10 | ±0.1 | [0, 0.3] | 0.05 s | 0.1 s |
| Fig. 11 / Table 3, PISR | `src/controllers/rt_controller_showerror.jl` | 1e4 | ±0.1 | ±0.3 | 0.02 s | 0.1 s |
| PHiL lab | `experiments/V-E_phil_lab_fig12_fig13/pisr_controller.jl` | 1e15 | ±3000 W | ±4000 W | 0.01 s | 0.1 s |

- **Fig. 10 overrides:** the values come from `experiments/V-D_online_resilience_fig10/controller_params.env`, which sets `P_WEIGHT`, `Q_WEIGHT`, `PENALTY_W`, `PLIMIT`, `QLIMIT`, `COMMON_V_LIMIT`/`VMAX` and `GLOB_MAXTIME`.
- **Online controller bounds:** the online PISR controllers optimize the change of the current setpoint, so ΔP may also be negative there, which releases curtailment.
- **Sensitivity benchmark:**
  - It is identified from `models/cil_longrun/train_data_complex.csv`.
  - The voltage sensitivities come from Δ-based least squares with ridge λ = 1e-6, with 3σ coefficient uncertainties.
  - The robust voltage bound is V + K·ΔS ± Σ\|ΔS\|·3σ_K.

## Timing measurement (Sec. V-C)

Full description with hardware and software versions: [SPEED_BENCHMARK.md](SPEED_BENCHMARK.md).

**Fair benchmark:** `experiments/V-C_offline_control_fig9/run.sh bench`

- Both voltage models are evaluated in-process, one input vector at a time, on the same 100 test inputs (`data/offline/test_data_complex_blockrand.csv`), on the same machine.
- Each input is warmed up, then timed repeatedly. The median per input is reported, and the median over the inputs is reported.
- **PISR, Julia** (`benchmark_pisr.jl`):
  - Evaluates all target equations of `models/offline_highvar` and returns max \|V\| with `eval_tree_array`. This is the path the controllers use.
  - Measured with BenchmarkTools `@belapsed`.
  - For reference it also times the high-level `MLJ.predict` on a one-row table.
- **PISR, Python** (in `benchmark_powerflow.py`): the same equations, exported from Julia, are evaluated as plain Python complex arithmetic. This removes the language difference to pandapower. The script checks that the values match the Julia results.
- **AC power flow** (`benchmark_powerflow.py`):
  - `pp.runpp` on the speed-baseline network (`src/services/powerflow_solver_api.py`), with the settings from the power-flow table: NR, flat start, 1e-7 MVA, 30 iterations, voltage angles, Q limits, numba.
  - Only the `runpp` call is timed, so there is no HTTP request and no network copy. pandapower's internal model build and result write are included, because every optimizer evaluation pays for them.
- **Output:** `results/V-C_offline_control_fig9/benchmark_summary.json` with the per-evaluation times and three speed-ups:
  - Python vs. Python
  - Julia PISR vs. pandapower
  - Julia `MLJ.predict` vs. pandapower

  It also records the \|V\|max deviation between PISR and the power flow.

**Optimization runs:** `run.sh` (PISR) and `run.sh speed`.

- They report the median optimization time per test sample, the number of objective evaluations and the time per evaluation.
- In `run.sh speed` every evaluation is a request to the local HTTP solver service. Its per-evaluation time therefore includes the request overhead and must not be used as the speed comparison; use `run.sh bench` for that.
