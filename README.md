# PISR voltage control on a low-voltage grid — reproduction package

Code, data and trained models for

> S. Eichhorn, A. Mohapatra, R. Tonkoski, **"Fast, Interpretable, and Agnostic: PISR Voltage Control on a Low-Voltage Grid"**

The repository follows the structure of the paper. Section III (the controller) is in `src/`.
Every result of Sections IV and V has one folder in `experiments/` and writes to the folder of the same name in `results/`.

| Paper | Content | Folder | Command | Runtime |
|---|---|---|---|---|
| Sec. III-B, Alg. 1 | PISR surrogate training | `src/pisr/` | used by the experiments below | |
| Sec. IV-A, Fig. 3 | training datasets | `experiments/IV-A_datasets_fig3` | `.venv/bin/python experiments/IV-A_datasets_fig3/plot_fig3.py` | seconds |
| Sec. V-A, Fig. 7 | correlated vs. high-variability training | `experiments/V-A_training_data_fig7` | `experiments/V-A_training_data_fig7/train.sh plot` | seconds |
| Sec. V-B, Fig. 8 | 10×10 cross-evaluation | `experiments/V-B_training_robustness_fig8` | `experiments/V-B_training_robustness_fig8/run.sh plot` | seconds |
| Sec. V-C, Fig. 9 + speed | offline optimization, PISR vs. power flow | `experiments/V-C_offline_control_fig9` | `experiments/V-C_offline_control_fig9/run.sh` | ~2 min |
| Sec. V-D, Fig. 10 | CIL: disturbance, topology change, noise | `experiments/V-D_online_resilience_fig10` | `experiments/V-D_online_resilience_fig10/run.sh` | ~5 min |
| Sec. V-D, Table 4, Fig. 12 | impedance increase, local Q(U) droop, least accurate model, offline optimum | `experiments/V-D_online_resilience_fig10` | `run_impedance_increase.sh run_06` (and `run_09`), then `offline_optimum.py` and `summarize_table4.py` | ~15 min per model |
| Sec. V-D, Fig. 11, Table 3 | long randomized CIL run | `experiments/V-D_long_run_fig11_table2` | `experiments/V-D_long_run_fig11_table2/run.sh fig11` | ~2 h |
| Sec. V-E, Figs. 13–14 | PHiL laboratory validation | `experiments/V-E_phil_lab_fig12_fig13` | `experiments/V-E_phil_lab_fig12_fig13/run.sh` (from the lab recordings) | seconds |

All commands are run from the repository root. Every solver, optimizer and training setting is listed in [docs/SETTINGS.md](docs/SETTINGS.md).

## Installation

Requirements: Linux, macOS or Windows; Julia 1.12 (tested 1.12.6), Python 3.12, `curl`, and bash for the `*.sh` drivers (Git Bash or WSL on Windows). The reference results in `results/` were recorded on Windows 11 (Intel Core i5-14500, 20 threads); see [docs/SPEED_BENCHMARK.md](docs/SPEED_BENCHMARK.md).

```bash
# Python (digital twin, power flow, plots)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Julia (PISR training, controllers, optimization); the Manifest pins SymbolicRegression 1.12.0
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

The trained models in `models/` are Julia-serialized MLJ machines. They load only with the package versions in `Manifest.toml`.

## Repository layout

```
data/
  grid/coses_matpower_source.xlsx    CoSES LV grid (buses, cables, loads) -> pandapower digital twin
  scenarios/                         AR(1) prosumer profiles, 3600 s at 1 s (generate_scenarios.jl)
    scenarios_random_high_var.csv    high-variability (decorrelated) profiles
    scenarios_correlated_pv.csv      correlated profiles (rho ~ 0.85)
    scenarios_correlated_pv_rt.csv   correlated profiles replayed by the real-time digital twin
  offline/                           offline training/test sets (Sec. IV-A, Figs. 3, 7, 9)
  lab/                               PHiL recordings of 2025-10-07 (NI VeriStand logger, measurement channels, Figs. 13, 14)
models/                              trained PISR models used in the paper
  offline_correlated/, offline_highvar/   Fig. 7 (offline_highvar is also the Fig. 9 model)
  cross_eval/runs/run_01..run_10/         Fig. 8: datasets collected over the REST API + one model each
  cil_longrun/                            PISR model of Figs. 10 and 11 / Table 3; its training data also
                                          identifies the sensitivity benchmark of Fig. 10
  phil_lab/                               PHiL model used in the lab + its training data collected on the hardware
src/
  grid/coses_net.py                  pandapower model of the CoSES grid (offline experiments)
  services/                          digital twin and power-flow services (FastAPI + pandapower)
  pisr/                              PISR training (Algorithm 1), data collection over the API, cross-evaluation
  controllers/                       PISR and sensitivity controllers (online), offline optimization (Fig. 9)
experiments/<section>_<figure>/      one driver per paper result
paper_figures/                       all paper figures as EPS, original file names (experiments/collect_paper_figures.sh)
results/                             reference outputs (figures, stats .json, data); large raw
                                     recordings as .csv.gz, logs not included. Re-running overwrites them.
```

### Components (Section III)

- **PISR surrogate:** `src/pisr/incremental_noslack.jl` for the offline models, and `src/pisr/incremental_rt.jl` for models trained on data collected over the API.
  - `MultitargetSRRegressor` with the binary operators `+ - * /`, the custom operators `line_current(V, S) = conj(S/V)` and `voltage_drop(I, Z) = I*Z` (`src/pisr/incwrap_ops.jl`; algebraically identical to `*` for complex arguments, so the effective operator set equals Algorithm 1 of the paper), `maxsize = 30` and `npopulations = 3 × (CPU threads − 2)`.
  - Training is incremental: 20 iterations first, then 100 more per step up to 1020. The test MAE is recorded after every step.
- **Self-learning over the REST API:** `src/pisr/collect_and_train_from_api.jl` perturbs the controllable inverters with random setpoints, records complex powers and voltages, and trains the model.
- **Online controller (cost function (1)):**
  - `src/controllers/rt_controller_showerror_routed.jl` (Fig. 10) and `rt_controller_showerror.jl` (Fig. 11) poll `/state`, predict `|V|max` with PISR and minimise `w_P ΣΔP² + ΣΔQ² + λ max(0, |V|max − 1.05)²` with differential evolution (BlackBoxOptim), using `w_P = 30`.
  - They post ΔP/ΔQ for the controllable buses 55, 57 and 64.
  - The control decision is open-loop: it uses only the PISR prediction from the power injections. Measured voltages are read for monitoring and logging only.
- **Sensitivity benchmark:** `src/controllers/rt_sensitivity_traindata_routed.jl`, a robust voltage-sensitivity controller in the style of Gupta et al.
  - It is identified from the same training data and minimises the same cost.
- **Digital twin** (`src/services/`):
  - `realtime_powerflow_service_routed_apicontrolled.py` is used for Figs. 8 and 10. It has a control router (none / sensitivity / pisr), breaker K4 (Q1 in the paper) and measurement noise.
  - `realtime_powerflow_service_with_noise.py` plus `_row_api.py` are used for Fig. 11 and Table 3.
  - All of them run a scenario loop at 1 Hz and a power-flow loop at 20 Hz.
  - `powerflow_solver_api.py` provides one AC power flow per request, as the baseline for the speed comparison.

## Reproducing the results

The trained models of the paper are shipped. The fast commands in the table regenerate every figure from them. The commands below re-run the expensive parts too.

### Sec. IV-A — datasets (Fig. 3)

```bash
.venv/bin/python experiments/IV-A_datasets_fig3/plot_fig3.py                # Fig. 3 from data/offline
.venv/bin/python experiments/IV-A_datasets_fig3/generate_datasets.py        # regenerate the offline sets (pandapower)
julia --project=. data/scenarios/generate_scenarios.jl [VARIANT=rt] [SEED=1] # regenerate AR(1) scenarios
```

- The training sets are deterministic given the scenario CSVs. `generate_datasets.py` reproduces `data/offline/train_data_complex*.csv` to 1e-12.
- The scenario profiles and the block randomization of the test set were drawn without a fixed seed. Use the shipped files for identical results.

### Sec. V-A — impact of the training data (Fig. 7)

```bash
experiments/V-A_training_data_fig7/train.sh plot   # Fig. 7 from the shipped training logs
experiments/V-A_training_data_fig7/train.sh        # retrain both models, then plot
```

Each model uses 100 randomly drawn rows (seed 0) of the 500-sample training set and is evaluated on `data/offline/test_data_complex_blockrand.csv`. Symbolic regression is stochastic, so retrained models differ slightly from the shipped ones.

### Sec. V-B — robustness of the training procedure (Fig. 8)

```bash
experiments/V-B_training_robustness_fig8/run.sh plot        # heatmaps + off-diagonal MAE ranges
experiments/V-B_training_robustness_fig8/run.sh crosseval   # re-evaluate the 10 shipped models on the 10 datasets
experiments/V-B_training_robustness_fig8/run.sh collect     # collect 10 new datasets + train 10 models (hours)
```

- `collect` starts the digital twin (measurement noise off, breaker K4 closed, i.e. ring topology, as at service start-up) and calls `batch_collect_and_train.sh`. Each run uses:
  - seed 4242 + 1000·i
  - perturbation magnitudes drawn from U(15, 80) kW/kvar
  - a random scenario start row
  - 50 training and 30 test samples
- The parameters of each paper run are in `models/cross_eval/manifest.csv` and `models/cross_eval/runs/run_XX/run_config.env`.
- The Fig. 8 datasets are therefore ring-topology data. The PISR model of Figs. 10 and 11 is `models/cil_longrun` instead (radial, see below).

### Sec. V-C — offline control and computational speed (Fig. 9)

```bash
experiments/V-C_offline_control_fig9/run.sh         # Fig. 9 + PISR timing
experiments/V-C_offline_control_fig9/run.sh bench   # fair per-evaluation benchmark PISR vs. pandapower runpp (in-process)
experiments/V-C_offline_control_fig9/run.sh speed   # identical optimization with pandapower runpp as inner model (via HTTP)
```

- `mpc_noslack.jl` optimizes every test sample with the `offline_highvar` model. `plot_fig9.py` then re-solves the AC power flow with the optimized setpoints to obtain the actual controlled voltage. As in the paper, the plotted bus is 57, the highest-voltage bus.
- `bench` is the speed comparison:
  - PISR and pandapower `runpp` are evaluated in-process, one input at a time, on the same 100 test inputs.
  - PISR is timed both in Julia, as used by the controllers, and as plain Python, the same runtime as pandapower.
  - The results are in `results/V-C_offline_control_fig9/benchmark_summary.json`. Baseline, warm-start, timing method and hardware are documented in [docs/SPEED_BENCHMARK.md](docs/SPEED_BENCHMARK.md).
- `run.sh` and `run.sh speed` print `[timing]` lines for the full optimization. `speed` calls the power flow over a local HTTP service, so its per-solve time includes request overhead and is not a fair comparison.

### Sec. V-D — online resilience (Fig. 10)

```bash
experiments/V-D_online_resilience_fig10/run.sh
```

- The driver starts the routed digital twin together with both controllers, then runs the sequence once per controller: Uncontrolled, then Robust Sensitivity, then PISR.
- Each sequence has three 5 s phases with a ΔP = −50 kW, ΔQ = −50 kvar step on bus 61 at t = 1–3 s:
  1. baseline, radial
  2. breaker Q1/K4 closed (ring)
  3. radial again with Gaussian measurement noise (σ = 5·10⁻³ p.u. per voltage component)
- PISR uses the radial-topology model `models/cil_longrun`; the sensitivity benchmark is identified from the same training data.
  - Its data (50 training / 20 test samples) was collected over the REST API from the digital twin in radial topology (breaker K4 open) with measurement noise off.
  - Retrain it from the shipped data (output in `results/training/`):
    `julia --project=. --threads auto src/pisr/incremental_rt.jl TRAIN_CSV=models/cil_longrun/train_data_complex.csv TEST_CSV=models/cil_longrun/test_data_complex_blockrand.csv SLACK_BUS_ID=7 MAX_TOTAL_ITER=1020`
- The controllers run at 10 Hz (`LOOP_PERIOD_S=0.1`, optimizer budget 0.05 s) and the plant at 20 Hz. Settings are in `controller_params.env`.
- To run the stack interactively, start `run_routed_stack.sh` and open `http://127.0.0.1:8012/ui`.

### Sec. V-D — impedance increase, local Q(U) droop and offline optimum (Table 4, Fig. 12)

```bash
experiments/V-D_online_resilience_fig10/run_impedance_increase.sh run_06
MODES="Uncontrolled,Robust Sensitivity,PISR" experiments/V-D_online_resilience_fig10/run_impedance_increase.sh run_09
.venv/bin/python experiments/V-D_online_resilience_fig10/offline_optimum.py results/V-D_impedance_increase/run_06/impedance_increase_recording.csv.gz
.venv/bin/python experiments/V-D_online_resilience_fig10/summarize_table4.py   # Table 4 values, settling times, cost vs. optimum
```

- Models identified on the ring topology (breaker Q1/K4 closed): `models/cross_eval/runs/run_06` (most accurate of Fig. 8) and `run_09` (least accurate). The sensitivity benchmark is identified from the same 50 training samples.
- Sequence: the ΔP = −50 kW, ΔQ = −50 kvar step on bus 61, held for 20 s, first on the ring, then after opening Q1 (unmodeled impedance increase); three repetitions.
- Local Q(U) droop (`QUDroop` in `sequence_impedance_increase.py`): each controllable inverter follows the standard characteristic on its own bus voltage (dead band 0.97–1.03 p.u., full reactive power at 0.93/1.07 p.u., first-order lag with 3τ = 10 s), Qmax = `QLIMIT` = 100 kvar, no active-power curtailment.
- Outputs: `results/V-D_impedance_increase/run_06|run_09/impedance_increase_summary.{csv,json}` (steady state over the last 3 s of each step), the recordings (`.csv.gz`) and Fig. 12 (`run_06/impedance_increase_comparison.eps`). `summarize_table4.py` writes `table4_means.csv` (Table 4), `table4_settling_times.csv` and `optimum/table4_optimum_comparison.csv` (cost (1) of each controller relative to the optimum).
- `offline_optimum.py` computes the minimum of cost (1) for the recorded operating points with the full network model (SLSQP on the AC power flow, same weights and bounds, |V| ≤ 1.05 p.u. at every energized LV bus) and writes `results/V-D_impedance_increase/optimum/optimum.json`.
- `diag_noise_repeats.py` repeats the Fig. 10 step 32 times (8 each without noise, with voltage and power noise, voltage noise only, power noise only) to count the short double-increment transients; result in `results/V-D_online_resilience_fig10/noise_repeats/`.
- Fig. 10 itself was re-recorded three times; recordings 2 and 3 and their steady-state comparison with the original recording are in `results/V-D_online_resilience_fig10/repeats/`.

### Sec. V-D — long randomized run (Fig. 11, Table 3)

```bash
experiments/V-D_long_run_fig11_table2/run.sh fig11    # 3600 s OFF + 3600 s ON  -> Fig. 11, Table 3 (~2 h)
experiments/V-D_long_run_fig11_table2/run.sh plot     # re-plot the shipped recordings
FIG11_DURATION=60 experiments/V-D_long_run_fig11_table2/run.sh fig11   # smoke test
```

- Setup: a −40 kW offset on uncontrolled bus 61, measurement noise off, breaker open. The scenario rows advance at 1 Hz, which gives one random P/Q event per second at the uncontrolled nodes.
- Fig. 11 replays the full 3600-row table starting at row 1000 (wrapping around) once without and once with the controller, about 72 000 samples per phase. Its statistics are in `results/V-D_long_run_fig11_table2/controller_comparison/comparison.stats.json`; the raw recordings are shipped as `off.csv.gz` / `on.csv.gz`.
- The shipped recording was made with the OFF and ON phases running concurrently on two identical digital-twin instances replaying the same rows (`sweep_off.json`, `sweep.json`); `run.sh fig11` records them one after the other on one instance.
- Table 3 of the paper is computed from this Fig. 11 comparison run. Its values are under `runs["PISR ON"]` in `comparison.stats.json`; `runs["Controller OFF"]` gives the uncontrolled baseline:
  - `high_sampling.voltage_error_pu.mean`, `voltage_rmse_pu` and `overvoltage_pu.max`
  - `violation_comparison.without_block.error_during_violations_pu`
  - `high_sampling.dP_mw` / `dQ_mvar` means

### Sec. V-E — PHiL laboratory validation (Figs. 13, 14)

```bash
experiments/V-E_phil_lab_fig12_fig13/run.sh   # Figs. 13 and 14 from data/lab/105setpoint*.csv
```

- Fig. 13 uses `105setpoint.csv`, logged at 100 Hz, with the controller enabled and disabled several times.
- Fig. 14 uses `105setpoint_high_speed.csv`, logged at 1 kHz, samples 1800–2100 around one activation.

The live experiment drives the CoSES laboratory hardware: an NI VeriStand gateway over OPC UA and EGSTON power amplifiers. It cannot be run without the lab. The scripts are the versions used on 2025-10-07:

| Step | Script |
|---|---|
| 1. REST bridge VeriStand ⇄ controller (OPC UA; pass the gateway with `--opcua-url` or `OPCUA_URL`) | `opcua_fastapi_service.py` |
| 2. Collect training data on the hardware and train PISR | `coses_train_from_api.jl` |
| 3. Retrain from a collected CSV | `incremental_training_coses.jl` (default `TRAIN_CSV=models/phil_lab/train_data_complex.csv`) |
| 4. PISR controller: 10 Hz, limit 1.05 p.u., buses SF1/SF4/SF6; enable/disable via `POST :8081/controller/{enable,disable}` | `pisr_controller.jl` (default `MODEL_PATH=models/phil_lab/final_model_iter1020.jls`) |
| 5. Figs. 13 and 14 from the logs | `plot_fig12_fig13.py` |

`models/phil_lab/` is the model used in the lab, trained on 100 radial-topology samples collected on the hardware. It was serialized on the lab PC with Julia 1.11.6 and an older SymbolicRegression release, so it does not load with the package versions in `Manifest.toml`. It is included as a record of the experiment.

## Reference results

These values come from running the repository on a Windows 11 workstation (Intel Core i5-14500, 20 threads). The second machine named in Sec. V-C (AMD Ryzen AI Max+ PRO 395, 32 threads, Linux) gave the timings in `results/V-C_offline_control_fig9/second_machine_amd_linux/`.

| Result | Value |
|---|---|
| Fig. 8 off-diagonal magnitude MAE | 3.5e-4 – 2.7e-3 p.u. |
| Fig. 9 controlled max \|V57\| | 1.0514 |
| Time per evaluation, PISR / `runpp` (in-process, `run.sh bench`) | Julia 4.39 µs / 17.4 ms, 3966×; Python vs. Python 563× (second machine: 2.2 µs / 4.67 ms, 2110×; 618×) |
| Fig. 9 optimization, objective evaluations per 100 ms decision (median) | 17 666 |
| Fig. 11 / Table 3, PISR ON: mean error / RMSE / max overvoltage | −0.00029 / 0.00230 / 0.01675 p.u. |
| Table 4, ring / radial max \|V\|, PISR model 06 | 1.0502 / 1.0604 p.u. (optimum 88.5 / 119.8 kvar) |
| Training time, `offline_highvar` settings (1020 iterations, 20 threads, compilation excluded) | 1734 s (`results/V-A_training_data_fig7/training_time.txt`) |

## Determinism

- Symbolic regression, the REST-API data collection, the online runs (wall-clock timing, asynchronous loops) and the measurement noise are not bit-reproducible.
- The offline pipeline, by contrast, is: datasets → Fig. 9 optimization → AC power-flow replay reproduces the paper data to within the optimizer's time budget, i.e. setpoint differences below 2·10⁻⁶ MW.
- The shipped models and datasets are the ones behind the paper figures.

## Data

- `data/grid/` describes the low-voltage grid of the CoSES laboratory at the Technical University of Munich.
- `data/lab/` contains the measurement channels recorded during the PHiL test on 2025-10-07: EGSTON amplifier P/Q setpoints and measurements, PV feed-in, and phase voltages.
- All other data in `data/` and `models/` was generated with the code in this repository.

## Citation

Citation details will be added once the paper is published.

PISR itself was introduced in: S. Eichhorn, A. Mohapatra, C. Goebel, "PISR: Physics-Informed Symbolic Regression for Predicting Power System Voltage", ACM e-Energy 2025, doi:10.1145/3679240.3734622.

## License

BSD 3-Clause, see [LICENSE](LICENSE). The code builds on permissively licensed packages only:

- Python: BSD/MIT/Apache-2.0/PSF.
- Julia: MIT and Apache-2.0 (SymbolicRegression.jl, DynamicExpressions.jl).

The lab bridge `experiments/V-E_phil_lab_fig12_fig13/opcua_fastapi_service.py` additionally imports `opcua` (python-opcua, LGPL-3.0). It is not bundled and is only needed in the laboratory.

## Contact

Sebastian Eichhorn, Chair of Electrical Power Transmission and Distribution, Technical University of Munich.

