# PISR voltage control on a low-voltage grid — reproduction package

Code, data and trained models for

> S. Eichhorn, A. Mohapatra, R. Tonkoski, **"Fast, Interpretable, and Agnostic: PISR Voltage Control on a Low-Voltage Grid"** — preprint: [doi:10.5281/zenodo.17395966](https://zenodo.org/records/17395966)

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
| Sec. V-D, Fig. 11, Table 2 | long randomized CIL run | `experiments/V-D_long_run_fig11_table2` | `experiments/V-D_long_run_fig11_table2/run.sh all` | ~2.5 h |
| Sec. V-E, Figs. 12–13 | PHiL laboratory validation | `experiments/V-E_phil_lab_fig12_fig13` | `experiments/V-E_phil_lab_fig12_fig13/run.sh` (from the lab recordings) | seconds |

All commands are run from the repository root. Every solver, optimizer and training setting is listed in [docs/SETTINGS.md](docs/SETTINGS.md).

## Installation

Requirements: Linux or macOS, Julia 1.12 (tested 1.12.6), Python 3.12, and `curl`.

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
  lab/                               PHiL recordings of 2025-10-07 (NI VeriStand logger, measurement channels, Figs. 12, 13)
models/                              trained PISR models used in the paper
  offline_correlated/, offline_highvar/   Fig. 7 (offline_highvar is also the Fig. 9 model)
  cross_eval/runs/run_01..run_10/         Fig. 8: datasets collected over the REST API + one model each
                                          (run_09 is the PISR model of Fig. 10)
  cil_longrun/                            Fig. 11 / Table 2 model; its training data also identifies
                                          the sensitivity benchmark of Fig. 10
  phil_lab/                               PHiL model used in the lab + its training data collected on the hardware
src/
  grid/coses_net.py                  pandapower model of the CoSES grid (offline experiments)
  services/                          digital twin and power-flow services (FastAPI + pandapower)
  pisr/                              PISR training (Algorithm 1), data collection over the API, cross-evaluation
  controllers/                       PISR and sensitivity controllers (online), offline optimization (Fig. 9)
experiments/<section>_<figure>/      one driver per paper result
results/                             reference outputs of the re-run (figures, stats .json, data); large raw
                                     recordings as .csv.gz, logs not included. Re-running overwrites them.
```

### Components (Section III)

- **PISR surrogate:** `src/pisr/incremental_noslack.jl` for the offline models, and `src/pisr/incremental_rt.jl` for models trained on data collected over the API.
  - `MultitargetSRRegressor` with the binary operators `+ - * /`, the custom operators `line_current(V, S) = conj(S/V)` and `voltage_drop(I, Z) = I*Z` (`src/pisr/incwrap_ops.jl`), `maxsize = 30` and `npopulations = 3 × (CPU threads − 2)`.
  - Training is incremental: 20 iterations first, then 100 more per step up to 1020. The test MAE is recorded after every step.
- **Self-learning over the REST API:** `src/pisr/collect_and_train_from_api.jl` perturbs the controllable inverters with random setpoints, records complex powers and voltages, and trains the model.
- **Online controller (cost function (1)):**
  - `src/controllers/rt_controller_showerror_routed.jl` (Fig. 10) and `rt_controller_showerror.jl` (Fig. 11) poll `/state`, predict `|V|max` with PISR and minimise `w_P ΣΔP² + ΣΔQ² + λ max(0, |V|max − 1.05)²` with differential evolution (BlackBoxOptim), using `w_P = 30`.
  - They post ΔP/ΔQ for the controllable buses 55, 57 and 64.
- **Sensitivity benchmark:** `src/controllers/rt_sensitivity_traindata_routed.jl`, a robust voltage-sensitivity controller in the style of [5], [26].
  - It is identified from the same training data and minimises the same cost.
- **Digital twin** (`src/services/`):
  - `realtime_powerflow_service_routed_apicontrolled.py` is used for Figs. 8 and 10. It has a control router (none / sensitivity / pisr), breaker K4 (Q1 in the paper) and measurement noise.
  - `realtime_powerflow_service_with_noise.py` plus `_row_api.py` are used for Fig. 11 and Table 2.
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

- `collect` starts the digital twin and calls `batch_collect_and_train.sh`. Each run uses:
  - seed 4242 + 1000·i
  - perturbation magnitudes drawn from U(15, 80) kW/kvar
  - a random scenario start row
  - 50 training and 30 test samples
- The parameters of each paper run are in `models/cross_eval/manifest.csv` and `models/cross_eval/runs/run_XX/run_config.env`.

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
  - The results are in `results/V-C_offline_control_fig9/benchmark_summary.json`. Details are in [docs/SETTINGS.md](docs/SETTINGS.md#timing-measurement-sec-v-c).
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
- The controllers run at 10 Hz (`LOOP_PERIOD_S=0.1`, optimizer budget 0.05 s) and the plant at 20 Hz. Settings are in `controller_params.env`.
- To run the stack interactively, start `run_routed_stack.sh` and open `http://127.0.0.1:8012/ui`.

### Sec. V-D — long randomized run (Fig. 11, Table 2)

```bash
experiments/V-D_long_run_fig11_table2/run.sh table2   # 1000 s PISR run -> Table 2   (~17 min)
experiments/V-D_long_run_fig11_table2/run.sh fig11    # 3600 s OFF + 3600 s ON  -> Fig. 11 (~2 h)
experiments/V-D_long_run_fig11_table2/run.sh plot     # re-plot existing recordings
T2_DURATION=60 FIG11_DURATION=60 experiments/V-D_long_run_fig11_table2/run.sh all   # smoke test
```

- Setup: a −40 kW offset on uncontrolled bus 61, measurement noise off, breaker open. The scenario rows advance at 1 Hz, which gives one random P/Q event per second at the uncontrolled nodes.
- Table 2 uses rows 799–1799. Its values are under `runs.PISR` in `results/V-D_long_run_fig11_table2/random_samples.stats.json`:
  - `high_sampling.voltage_error_pu.mean`, `voltage_rmse_pu` and `overvoltage_pu.max`
  - `violation_comparison.without_block.error_during_violations_pu`
  - `high_sampling.dP_mw` / `dQ_mvar` means
- Fig. 11 replays the full 3600-row table starting at row 1000 (wrapping around) once without and once with the controller.

### Sec. V-E — PHiL laboratory validation (Figs. 12, 13)

```bash
experiments/V-E_phil_lab_fig12_fig13/run.sh   # Figs. 12 and 13 from data/lab/105setpoint*.csv
```

- Fig. 12 uses `105setpoint.csv`, logged at 100 Hz, with the controller enabled and disabled several times.
- Fig. 13 uses `105setpoint_high_speed.csv`, logged at 1 kHz, samples 1800–2100 around one activation.

The live experiment drives the CoSES laboratory hardware: an NI VeriStand gateway over OPC UA and EGSTON power amplifiers. It cannot be run without the lab. The scripts are the versions used on 2025-10-07:

| Step | Script |
|---|---|
| 1. REST bridge VeriStand ⇄ controller (OPC UA; pass the gateway with `--opcua-url` or `OPCUA_URL`) | `opcua_fastapi_service.py` |
| 2. Collect training data on the hardware and train PISR | `coses_train_from_api.jl` |
| 3. Retrain from a collected CSV | `incremental_training_coses.jl` (default `TRAIN_CSV=models/phil_lab/train_data_complex.csv`) |
| 4. PISR controller: 10 Hz, limit 1.05 p.u., buses SF1/SF4/SF6; enable/disable via `POST :8081/controller/{enable,disable}` | `pisr_controller.jl` (default `MODEL_PATH=models/phil_lab/final_model_iter1020.jls`) |
| 5. Figs. 12 and 13 from the logs | `plot_fig12_fig13.py` |

`models/phil_lab/` is the model used in the lab, trained on 100 radial-topology samples collected on the hardware. It was serialized on the lab PC with Julia 1.11.6 and an older SymbolicRegression release, so it does not load with the package versions in `Manifest.toml`. It is included as a record of the experiment.

## Reference re-run

These values come from re-running the repository on a 32-thread Linux workstation.

| Result | Paper | Re-run |
|---|---|---|
| Fig. 8 off-diagonal magnitude MAE | 3.5e-4 – 2.7e-3 p.u. | identical (all 100 pairs) |
| Fig. 9 controlled max \|V57\| | 1.0514 | 1.0514 |
| Time per evaluation, PISR / `runpp` (in-process, `run.sh bench`) | ~6 µs / ~10 ms, ~2000× | Julia 4.6 µs / 11.9 ms, 2577×; Python vs. Python 19 µs / 11.9 ms, 627× |
| Table 2 mean voltage error / RMSE | −0.00043 / 0.00180 p.u. | −0.00046 / 0.00183 p.u. |
| Table 2 max overvoltage | 0.01055 p.u. | 0.01180 p.u. |
| Table 2 mean ΣΔP / ΣΔQ | 0.58 kW / 61.8 kvar | 0.45 kW / 62.0 kvar |

## Determinism

- Symbolic regression, the REST-API data collection, the online runs (wall-clock timing, asynchronous loops) and the measurement noise are not bit-reproducible.
- The offline pipeline, by contrast, is: datasets → Fig. 9 optimization → AC power-flow replay reproduces the paper data to within the optimizer's time budget, i.e. setpoint differences below 2·10⁻⁶ MW.
- The shipped models and datasets are the ones behind the paper figures.

## Data

- `data/grid/` describes the low-voltage grid of the CoSES laboratory at the Technical University of Munich [24].
- `data/lab/` contains the measurement channels recorded during the PHiL test on 2025-10-07: EGSTON amplifier P/Q setpoints and measurements, PV feed-in, and phase voltages.
- All other data in `data/` and `models/` was generated with the code in this repository.

## Citation

Until the journal version is published, please cite the preprint:

> S. Eichhorn, A. Mohapatra, R. Tonkoski, "Fast, Interpretable, and Agnostic - PISR Voltage Control on a Low-Voltage Grid", preprint, Zenodo, 2025. doi:[10.5281/zenodo.17395966](https://zenodo.org/records/17395966)

```bibtex
@misc{eichhorn2025pisrcontrol,
  author    = {Eichhorn, Sebastian and Mohapatra, Anurag and Tonkoski, Reinaldo},
  title     = {Fast, Interpretable, and Agnostic - {PISR} Voltage Control on a Low-Voltage Grid},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17395966},
  url       = {https://zenodo.org/records/17395966},
  note      = {Preprint}
}
```

PISR itself was introduced in: S. Eichhorn, A. Mohapatra, C. Goebel, "PISR: Physics-Informed Symbolic Regression for Predicting Power System Voltage", ACM e-Energy 2025, doi:10.1145/3679240.3734622.

## License

BSD 3-Clause, see [LICENSE](LICENSE). The code builds on permissively licensed packages only:

- Python: BSD/MIT/Apache-2.0/PSF.
- Julia: MIT and Apache-2.0 (SymbolicRegression.jl, DynamicExpressions.jl).

The lab bridge `experiments/V-E_phil_lab_fig12_fig13/opcua_fastapi_service.py` additionally imports `opcua` (python-opcua, LGPL-3.0). It is not bundled and is only needed in the laboratory.

## Contact

Sebastian Eichhorn, Chair of Electrical Power Transmission and Distribution, Technical University of Munich.

