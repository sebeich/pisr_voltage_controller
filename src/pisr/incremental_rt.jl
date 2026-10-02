#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
Incremental symbolic regression training with periodic evaluation and early stopping.

Combines functionality of `approximate_from_powerflow_csv.jl` (training) and
`evaluate_model_from_complex_csv.jl` (evaluation) into reusable functions.

Workflow:
 1. Load complex training and test CSVs (Python-style complex tokens `(a+bj)`).
 2. Build features (slack voltage + all complex powers) and targets (all other voltages).
 3. Train a `MultitargetSRRegressor` for an initial `INITIAL_ITER` iterations.
 4. Evaluate on the test set computing per-bus magnitude/angle MAE and Max Error.
 5. If worst magnitude max error < THRESHOLD, stop early. Otherwise, increase
    `model.niterations += STEP_ITER`, call `fit!(mach)` to continue training, and repeat.
 6. Record metrics trajectory to CSV and produce plots of error development.
 7. For every evaluation stage, save a plot of the worst bus (magnitude & angle time series).

Stopping Criterion:
  * `worst_mag_max < THRESHOLD`
    (Where `worst_mag_max` = maximum across non-slack buses of per-bus maximum |Δ|V||.)

Outputs (stored in a timestamped subdirectory):
  * `metrics_progress.csv` : iteration-wise aggregate metrics.
  * `metrics_progress_plot.svg` : curves of mean/worst MAE & Max errors vs iterations.
  * `worst_bus_iterXXXX.svg` : worst bus timeseries magnitude + angle per eval stage.
  * `model_iterXXXX.jls` : serialized machine snapshot each stage.
  * `final_model.jls` : final machine (symlink or copy of last snapshot).
  * `final_metrics.txt` : human-readable summary.

Environment / CLI overrides (either ENV vars or `--key=value` args):
  TRAIN_CSV          (default: data/offline/train_data_complex.csv)
  TEST_CSV           (default: data/offline/test_data_complex_blockrand.csv)
  SLACK_BUS_ID       (default: 65)
  INITIAL_ITER       (default: 10)
  STEP_ITER          (default: 5)
  MAX_TOTAL_ITER     (default: 120)
  THRESHOLD          (default: 0.01)
  MAX_SIZE           (default: 30)
  NP_FACTOR          (default: 3; sets npopulations = workers*NP_FACTOR)
  MULTIPROCESS       (default: false)

Example:
  julia incremental_train_evaluate.jl INITIAL_ITER=15 STEP_ITER=5 THRESHOLD=0.005

"""
# Run from the repository root: julia --project=. --threads auto src/pisr/incremental_rt.jl KEY=VALUE ...
using DataFrames
using CSV
using SymbolicRegression
using MLJ
using Serialization
using Statistics
using LinearAlgebra
using Printf
using Plots
using Random
using Dates

# ---------------- Utility: Argument & ENV retrieval ----------------
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function parse_kv_args!(dict)
    for a in ARGS
        if occursin('=', a)
            k,v = split(a, '=', limit=2)
            dict[k] = v
        end
    end
    return dict
end

function getenv_or(args::Dict, key::String, default)
    haskey(args, key) && return args[key]
    get(ENV, key, default)
end

_raw_args = parse_kv_args!(Dict{String,String}())

TRAIN_COMPLEX_CSV = normpath(joinpath(REPO_ROOT, getenv_or(_raw_args, "TRAIN_CSV", "data/offline/train_data_complex.csv")))
TEST_COMPLEX_CSV  = normpath(joinpath(REPO_ROOT, getenv_or(_raw_args, "TEST_CSV",  "data/offline/test_data_complex_blockrand.csv")))
SLACK_BUS_ID      = parse(Int, getenv_or(_raw_args, "SLACK_BUS_ID", "66"))
INITIAL_ITER      = parse(Int, getenv_or(_raw_args, "INITIAL_ITER", "20"))
STEP_ITER         = parse(Int, getenv_or(_raw_args, "STEP_ITER", "100"))
MAX_TOTAL_ITER    = parse(Int, getenv_or(_raw_args, "MAX_TOTAL_ITER", "1000"))
THRESHOLD         = parse(Float64, getenv_or(_raw_args, "THRESHOLD", "0.001"))
MAX_SIZE          = parse(Int, getenv_or(_raw_args, "MAX_SIZE", "30"))
NP_FACTOR         = parse(Int, getenv_or(_raw_args, "NP_FACTOR", "3"))
MULTIPROCESS      = lowercase(getenv_or(_raw_args, "MULTIPROCESS", "false")) in ("1","true","yes")

# New optional controls for limiting / sampling training rows
TRAIN_LIMIT       = parse(Int, getenv_or(_raw_args, "TRAIN_LIMIT", "100"))  # 0 => use all
TRAIN_SAMPLE_MODE = lowercase(getenv_or(_raw_args, "TRAIN_SAMPLE_MODE", "random"))  # first|random
TRAIN_SEED        = parse(Int, getenv_or(_raw_args, "TRAIN_SEED", "0"))
SAVE_SNAPSHOTS    = lowercase(getenv_or(_raw_args, "SAVE_SNAPSHOTS", "false")) in ("1","true","yes")  # if true keep intermediate models

@info "Config" TRAIN_COMPLEX_CSV TEST_COMPLEX_CSV SLACK_BUS_ID INITIAL_ITER STEP_ITER MAX_TOTAL_ITER THRESHOLD MAX_SIZE NP_FACTOR MULTIPROCESS TRAIN_LIMIT TRAIN_SAMPLE_MODE TRAIN_SEED SAVE_SNAPSHOTS

isfile(TRAIN_COMPLEX_CSV) || error("Missing training complex CSV: " * TRAIN_COMPLEX_CSV)
if !isfile(TEST_COMPLEX_CSV)
    @warn "Test CSV missing; will use training data for evaluation" TEST_COMPLEX_CSV
    TEST_COMPLEX_CSV = TRAIN_COMPLEX_CSV
end

# ---------------- Complex CSV Loader (shared with existing scripts) ----------------
function load_complex_raw(path::AbstractString)
    txt = read(path, String)
    lines = split(txt, '\n')
    isempty(lines) && error("Empty file: $path")
    header_line = strip(lines[1])
    header = Symbol.(split(header_line, ','))
    body = join(lines[2:end], "\n")
    token_pattern = r"\([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[+-](?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?j\)"
    tokens = [m.match for m in eachmatch(token_pattern, body)]
    n_cols = length(header)
    length(tokens) % n_cols == 0 || error("Token count $(length(tokens)) not divisible by columns $n_cols")
    n_rows = length(tokens) ÷ n_cols
    rows = Vector{Vector{String}}(undef, n_rows)
    for r in 1:n_rows
        rows[r] = tokens[(n_cols*(r-1)+1):(n_cols*r)]
    end
    return header, rows
end

function parse_py_complex(str::AbstractString)
    s = replace(strip(str), r"\s+" => "")
    startswith(s, "(") && endswith(s, ")") && (s = s[2:end-1])
    m = match(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([+-](?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)j$", s)
    m === nothing && error("Cannot parse complex token: " * str)
    real_str, imag_str = m.captures
    return ComplexF64(parse(Float64, real_str), parse(Float64, imag_str))
end

function load_complex_df(path::AbstractString)
    header, raw_rows = load_complex_raw(path)
    df = DataFrame()
    n_samples = length(raw_rows)
    for (j,h) in enumerate(header)
        col_data = Vector{ComplexF64}(undef, n_samples)
        for i in 1:n_samples
            col_data[i] = parse_py_complex(raw_rows[i][j])
        end
        df[!,h] = col_data
    end
    return df
end

# ---------------- Feature / Target Preparation ----------------
struct DatasetFT
    X::NamedTuple
    y::NamedTuple
    slack_bus::Int
    target_syms::Vector{Symbol}
    input_syms::Vector{Symbol}
end

function prepare_features_targets(df::DataFrame, slack_bus::Int)
    voltage_cols = [c for c in names(df) if occursin(r"^V\d+_complex$", String(c))]
    power_cols   = [c for c in names(df) if occursin(r"^S\d+_complex$", String(c))]
    @assert !isempty(voltage_cols) "No voltage columns found"
    @assert !isempty(power_cols)   "No power columns found"
    voltage_bus_ids = sort(parse.(Int, match.(Ref(r"^V(\d+)_complex$"), String.(voltage_cols)) .|> x->x.captures[1]))
    # Determine load-connected buses from available S*_complex columns
    load_bus_ids = sort(parse.(Int, match.(Ref(r"^S(\d+)_complex$"), String.(power_cols)) .|> x->x.captures[1]))
    @assert slack_bus in voltage_bus_ids "Slack bus $(slack_bus) missing in dataset"
    slack_col = Symbol(@sprintf("V%d_complex", slack_bus))
    # Only predict voltages at buses that have a load (S column), excluding slack
    target_voltage_cols = Symbol[]

    for v in Symbol.(voltage_cols)
        m = match(r"^V(\d+)_complex$", String(v))
        bus_id = parse(Int, m.captures[1])
        if bus_id != slack_bus && (bus_id in load_bus_ids)
            push!(target_voltage_cols, v)
        end
    end
    target_voltage_cols = [Symbol("V64_complex"), Symbol("V57_complex"), Symbol("V55_complex"), Symbol("V61_complex")]
    @assert !isempty(target_voltage_cols) "No non-slack load-connected voltage columns found"
    input_syms = [Symbol.(power_cols);]
    input_syms = [Symbol("S64_complex"), Symbol("S57_complex"), Symbol("S55_complex"), Symbol("S61_complex")]

    inputs_df = DataFrame()
    for c in Symbol.(power_cols); inputs_df[!,c] = df[!,c]; end
    targets_df = DataFrame()
    for c in target_voltage_cols; targets_df[!,c] = df[!,c]; end
    X = NamedTuple{Tuple(input_syms)}(Tuple(inputs_df[!,c] for c in input_syms))
    y = NamedTuple{Tuple(target_voltage_cols)}(Tuple(targets_df[!,c] for c in target_voltage_cols))
    return DatasetFT(X, y, slack_bus, target_voltage_cols, input_syms)
end

# ---------------- Custom Operators (must exist before model usage) ----------------
line_current(V_send::Complex, S::Complex) = conj(S / V_send)
voltage_drop(I::Complex, Z::Complex) = I * Z

# ---------------- Metrics ----------------
struct EvalMetrics
    iteration::Int
    mean_mag_mae::Float64
    worst_mag_max::Float64
    mean_ang_mae::Float64
    worst_ang_max::Float64
    worst_bus::Int
end

get_mag_angle(z) = (abs(z), angle(z))

function compute_eval_metrics(y_true::NamedTuple, y_pred::NamedTuple, iteration::Int)
    bus_metrics = DataFrame(bus=Int[], mag_mae=Float64[], mag_max=Float64[], ang_mae=Float64[], ang_max=Float64[])
    for sym in keys(y_true)
        actual = y_true[sym]; pred = y_pred[sym]
        actual_ma = get_mag_angle.(actual); pred_ma = get_mag_angle.(pred)
        act_mag = first.(actual_ma); pred_mag = first.(pred_ma)
        act_ang = last.(actual_ma); pred_ang = last.(pred_ma)
        mag_errs = abs.(act_mag .- pred_mag)
        ang_errs = abs.(act_ang .- pred_ang)
        push!(bus_metrics, (parse(Int, match(r"^V(\d+)_complex$", String(sym)).captures[1]), mean(mag_errs), maximum(mag_errs), mean(ang_errs), maximum(ang_errs)))
    end
    worst_row = first(sort(bus_metrics, :mag_max, rev=true))
    mean_mag_mae = mean(bus_metrics.mag_mae)
    worst_mag_max = worst_row.mag_max
    mean_ang_mae = mean(bus_metrics.ang_mae)
    worst_ang_max = maximum(bus_metrics.ang_max)
    return EvalMetrics(iteration, mean_mag_mae, worst_mag_max, mean_ang_mae, worst_ang_max, worst_row.bus), bus_metrics
end

function plot_worst_bus!(output_dir::String, y_true::NamedTuple, y_pred::NamedTuple, worst_bus::Int, iteration::Int)
    sym = Symbol(@sprintf("V%d_complex", worst_bus))
    actual = y_true[sym]; pred = y_pred[sym]
    actual_ma = get_mag_angle.(actual); pred_ma = get_mag_angle.(pred)
    act_mag = first.(actual_ma); pred_mag = first.(pred_ma)
    act_ang = last.(actual_ma); pred_ang = last.(pred_ma)
    p1 = plot(act_mag; label="Actual", title=@sprintf("Worst Bus %d |V| iter %d", worst_bus, iteration), xlabel="Sample", ylabel="|V|")
    plot!(p1, pred_mag; label="Pred")
    p2 = plot(act_ang; label="Actual", title=@sprintf("Worst Bus %d angle iter %d", worst_bus, iteration), xlabel="Sample", ylabel="rad")
    plot!(p2, pred_ang; label="Pred")
    out_path = joinpath(output_dir, @sprintf("worst_bus_iter%04d.svg", iteration))
    savefig(plot(p1,p2; layout=(2,1), size=(700,600)), out_path)
    return out_path
end

function plot_progress(metrics_vec::Vector{EvalMetrics}, output_dir::String)
    its = [m.iteration for m in metrics_vec]
    mean_mag_mae = [m.mean_mag_mae for m in metrics_vec]
    worst_mag_max = [m.worst_mag_max for m in metrics_vec]
    mean_ang_mae = [m.mean_ang_mae for m in metrics_vec]
    worst_ang_max = [m.worst_ang_max for m in metrics_vec]
    p1 = plot(its, mean_mag_mae; label="Mean Mag MAE", xlabel="Iterations", ylabel="Error", title="Magnitude Errors")
    plot!(p1, its, worst_mag_max; label="Worst Mag Max")
    p2 = plot(its, mean_ang_mae; label="Mean Ang MAE", xlabel="Iterations", ylabel="Error", title="Angle Errors")
    plot!(p2, its, worst_ang_max; label="Worst Ang Max")
    savefig(plot(p1,p2; layout=(1,2), size=(1200,450)), joinpath(output_dir, "metrics_progress_plot.svg"))
end

# ---------------- Training Sequence ----------------
function incremental_train(train_df::DataFrame, test_df::DataFrame; slack_bus::Int, initial_iter::Int, step_iter::Int, max_total_iter::Int, threshold::Float64, max_size::Int, np_factor::Int, multiprocess::Bool, output_dir::String, save_snapshots::Bool=false)
    mkpath(output_dir)
    # Prepare datasets
    d_train = prepare_features_targets(train_df, slack_bus)
    d_test  = prepare_features_targets(test_df, slack_bus)
    @assert d_train.target_syms == d_test.target_syms "Train/Test target symbol mismatch"
    usedprocessors = max(1, Sys.CPU_THREADS - 2)
    function elementwise_outlier_loss(prediction, target;
                                    eps=1e-9, p=4, k=3.0)
        # Relative error
        rel = abs(prediction - target) / (abs(target) + eps)
        # Base (RMSE-like) term + higher-power term amplifying outliers
        return rel^2 + k * rel^p
    end
    model = MultitargetSRRegressor(
        binary_operators = [+, -, *, /, line_current, voltage_drop],
        npopulations = usedprocessors * np_factor,
        niterations = initial_iter,
        maxsize = max_size,
        #elementwise_loss = (ŷ, y) -> elementwise_outlier_loss(ŷ, y; p=6, k=4.0),
        parallelism = multiprocess ? :multiprocessing : :multithreading,
        numprocs = multiprocess ? usedprocessors : nothing,
        save_to_file = false,
    )
    mach = machine(model, d_train.X, d_train.y)
    total_iter = initial_iter
    metrics_vec = EvalMetrics[]
    metrics_detail_dfs = Dict{Int,DataFrame}()
    @info "Initial fit" total_iter
    @time fit!(mach)
    y_pred_test = predict(mach, d_test.X)
    em, bus_df = compute_eval_metrics(d_test.y, y_pred_test, total_iter)
    push!(metrics_vec, em); metrics_detail_dfs[total_iter] = bus_df
    plot_worst_bus!(output_dir, d_test.y, y_pred_test, em.worst_bus, total_iter)
    if save_snapshots
        serialize(joinpath(output_dir, @sprintf("model_iter%04d.jls", total_iter)), mach)
    end
    while em.worst_mag_max > threshold && total_iter < max_total_iter
        model.niterations = total_iter + step_iter
        total_iter = model.niterations
        @info "Continuing training" total_iter
        fit!(mach)  # updates existing populations
        y_pred_test = predict(mach, d_test.X)
        em, bus_df = compute_eval_metrics(d_test.y, y_pred_test, total_iter)
        push!(metrics_vec, em); metrics_detail_dfs[total_iter] = bus_df
        plot_worst_bus!(output_dir, d_test.y, y_pred_test, em.worst_bus, total_iter)
        if save_snapshots
            serialize(joinpath(output_dir, @sprintf("model_iter%04d.jls", total_iter)), mach)
        end
    end
    # Progress artifacts
    prog_df = DataFrame(iteration=[m.iteration for m in metrics_vec], mean_mag_mae=[m.mean_mag_mae for m in metrics_vec], worst_mag_max=[m.worst_mag_max for m in metrics_vec], mean_ang_mae=[m.mean_ang_mae for m in metrics_vec], worst_ang_max=[m.worst_ang_max for m in metrics_vec], worst_bus=[m.worst_bus for m in metrics_vec])
    CSV.write(joinpath(output_dir, "metrics_progress.csv"), prog_df)
    plot_progress(metrics_vec, output_dir)
    # Final summary
    open(joinpath(output_dir, "final_metrics.txt"), "w") do io
        println(io, "Incremental Training Summary")
        println(io, "============================")
        println(io, @sprintf("Slack bus: %d", slack_bus))
        println(io, @sprintf("Final Iterations: %d", metrics_vec[end].iteration))
        println(io, @sprintf("Final Mean Mag MAE: %.6f", metrics_vec[end].mean_mag_mae))
        println(io, @sprintf("Final Worst Mag Max: %.6f", metrics_vec[end].worst_mag_max))
        println(io, @sprintf("Final Mean Ang MAE: %.6f", metrics_vec[end].mean_ang_mae))
        println(io, @sprintf("Final Worst Ang Max: %.6f", metrics_vec[end].worst_ang_max))
        println(io, @sprintf("Worst Bus Final: %d", metrics_vec[end].worst_bus))
        println(io, @sprintf("Threshold: %.6f  (Reached: %s)", threshold, metrics_vec[end].worst_mag_max <= threshold))
    end
    final_model_path = joinpath(output_dir, @sprintf("final_model_iter%04d.jls", metrics_vec[end].iteration))
    serialize(final_model_path, mach)

    # ---- Compute SR-based voltage magnitude prediction uncertainties per influence factor ----
    # For each target voltage and each input (S_j), approximate the per-unit uncertainty
    # in |V| prediction attributable to ΔP_j and ΔQ_j using ensemble finite-difference slopes
    # across the training set and the top-K SR equations.
    try
        # Config
        local SR_Z = try parse(Float64, get(ENV, "SR_Z", "3.0")) catch; 3.0 end
        local SR_ENSEMBLE_K = try parse(Int, get(ENV, "SR_ENSEMBLE_K", "5")) catch; 5 end
        local UNC_SAMPLES = try parse(Int, get(ENV, "SR_UNC_SAMPLES", "300")) catch; 300 end
        local DELTA_STEP = try parse(Float64, get(ENV, "DELTA_STEP", "1e-3")) catch; 1e-3 end

        # SR internals
        local rep = report(mach)
        local eq_groups = rep.equations             # Vector{Vector{ExprNode}} per target
        local best_idx = rep.best_idx
        local n_targets = length(eq_groups)
        @assert n_targets == length(d_train.target_syms) "Target size mismatch"
        # Build ensembles
        local ensembles = Vector{Vector{Any}}(undef, n_targets)
        for i in 1:n_targets
            eqs = eq_groups[i]
            ensembles[i] = eqs[1:min(SR_ENSEMBLE_K, length(eqs))]
        end

        # Feature order and buffers
        local input_syms = d_train.input_syms
        local N_FEATURES = length(input_syms)
        local _Xscratch = Vector{ComplexF64}(undef, N_FEATURES)
        local _Xcol = reshape(_Xscratch, N_FEATURES, 1)
        local function _fill_row!(row_idx)
            @inbounds for (k,s) in enumerate(input_syms)
                _Xscratch[k] = ComplexF64(d_train.X[s][row_idx])
            end
            return _Xcol
        end

        local function _pred_abs_for_target!(ti)
            # returns abs values for each eq in ensemble for target ti at current _Xcol
            eqs = ensembles[ti]
            vals = Vector{Float64}(undef, length(eqs))
            @inbounds for k in eachindex(eqs)
                y = eval_tree_array(eqs[k], _Xcol)
                yr = y isa Tuple ? y[1] : y
                v = yr isa AbstractArray ? yr[1] : yr
                vals[k] = abs(ComplexF64(v))
            end
            return vals
        end

        # Sampling rows
        local N = nrow(train_df)
        local R = min(UNC_SAMPLES, N)
        local rows = collect(1:R)

        # Accumulators per (target, input)
        # We gather all per-row, per-eq finite-difference slopes and take std across all.
        local dP_slopes = [Float64[] for _ in 1:n_targets, __ in 1:length(input_syms)]
        local dQ_slopes = [Float64[] for _ in 1:n_targets, __ in 1:length(input_syms)]

        for r in rows
            _fill_row!(r)
            # baseline abs for each target and eq
            local base_abs = [ _pred_abs_for_target!(ti) for ti in 1:n_targets ]
            # For each input, perturb P and Q separately
            for (j, s) in enumerate(input_syms)
                # P perturb
                local orig = _Xscratch[j]
                _Xscratch[j] = ComplexF64(real(orig) + DELTA_STEP, imag(orig))
                local abs_plusP = [ _pred_abs_for_target!(ti) for ti in 1:n_targets ]
                _Xscratch[j] = orig
                # Q perturb
                _Xscratch[j] = ComplexF64(real(orig), imag(orig) + DELTA_STEP)
                local abs_plusQ = [ _pred_abs_for_target!(ti) for ti in 1:n_targets ]
                _Xscratch[j] = orig

                # collect per-target slopes per eq
                for ti in 1:n_targets
                    @inbounds begin
                        for k in eachindex(base_abs[ti])
                            push!(dP_slopes[ti, j], (abs_plusP[ti][k] - base_abs[ti][k]) / DELTA_STEP)
                            push!(dQ_slopes[ti, j], (abs_plusQ[ti][k] - base_abs[ti][k]) / DELTA_STEP)
                        end
                    end
                end
            end
        end

        # Compute per-unit uncertainty = Z * std(slopes) for each (target,input)
        local out = DataFrame(
            target_voltage = String[],
            input_power    = String[],
            dP_unit_sigma  = Float64[],
            dQ_unit_sigma  = Float64[],
            dP_unit_margin = Float64[],  # = Z * sigma
            dQ_unit_margin = Float64[],
            Z = Float64[],
            delta_step = Float64[],
        )
        for ti in 1:n_targets
            tname = String(d_train.target_syms[ti])
            for j in 1:length(input_syms)
                iname = String(input_syms[j])
                σp = std(dP_slopes[ti, j])
                σq = std(dQ_slopes[ti, j])
                push!(out, (tname, iname, σp, σq, SR_Z*σp, SR_Z*σq, SR_Z, DELTA_STEP))
            end
        end
        local outf = joinpath(output_dir, "sr_prediction_uncertainties_by_factor.csv")
        CSV.write(outf, out)
        @info "Exported SR prediction uncertainties by factor" file=outf rows=nrow(out)
    catch err
        @warn "Failed to compute/export SR prediction uncertainties by factor" err
    end

    return mach, metrics_vec, metrics_detail_dfs
end

# Convenience wrapper for REPL use (pass explicit paths & params)
function run_incremental_training(;train_csv=TRAIN_COMPLEX_CSV, test_csv=TEST_COMPLEX_CSV, slack_bus=SLACK_BUS_ID, initial_iter=INITIAL_ITER, step_iter=STEP_ITER, max_total_iter=MAX_TOTAL_ITER, threshold=THRESHOLD, max_size=MAX_SIZE, np_factor=NP_FACTOR, multiprocess=MULTIPROCESS, train_limit=TRAIN_LIMIT, train_sample_mode=TRAIN_SAMPLE_MODE, train_seed=TRAIN_SEED, save_snapshots=SAVE_SNAPSHOTS, output_dir=nothing)
    train_df = load_complex_df(train_csv)
    if train_limit > 0 && train_limit < nrow(train_df)
        if train_sample_mode == "random"
            Random.seed!(train_seed)
            idx = randperm(nrow(train_df))[1:train_limit]
            train_df = train_df[idx, :]
        else
            train_df = train_df[1:train_limit, :]
        end
    end
    test_df = train_csv == test_csv ? train_df : load_complex_df(test_csv)
    if output_dir === nothing
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        output_dir = joinpath(REPO_ROOT, "results", "training", @sprintf("incremental_training_%s", timestamp))
    end
    return incremental_train(train_df, test_df; slack_bus=slack_bus, initial_iter=initial_iter, step_iter=step_iter, max_total_iter=max_total_iter, threshold=threshold, max_size=max_size, np_factor=np_factor, multiprocess=multiprocess, output_dir=output_dir, save_snapshots=save_snapshots)
end

# ---------------- Main ----------------
function main()
    train_df = load_complex_df(TRAIN_COMPLEX_CSV)
    n_total = nrow(train_df)
    if TRAIN_LIMIT > 0 && TRAIN_LIMIT < n_total
        if TRAIN_SAMPLE_MODE == "first"
            train_df = train_df[1:TRAIN_LIMIT, :]
        elseif TRAIN_SAMPLE_MODE == "random"
            Random.seed!(TRAIN_SEED)
            idx = randperm(n_total)[1:TRAIN_LIMIT]
            train_df = train_df[idx, :]
        else
            @warn "Unknown TRAIN_SAMPLE_MODE=$(TRAIN_SAMPLE_MODE); using 'first'"
            train_df = train_df[1:TRAIN_LIMIT, :]
        end
        @info "Subsampled training rows" original=n_total used=nrow(train_df) mode=TRAIN_SAMPLE_MODE seed=TRAIN_SEED
    else
        @info "Using all training rows" count=n_total
    end
    test_df  = TRAIN_COMPLEX_CSV == TEST_COMPLEX_CSV ? train_df : load_complex_df(TEST_COMPLEX_CSV)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out_dir = joinpath(REPO_ROOT, "results", "training", @sprintf("incremental_training_%s", timestamp))
    mach, metrics_vec, _ = incremental_train(train_df, test_df; slack_bus=SLACK_BUS_ID, initial_iter=INITIAL_ITER, step_iter=STEP_ITER, max_total_iter=MAX_TOTAL_ITER, threshold=THRESHOLD, max_size=MAX_SIZE, np_factor=NP_FACTOR, multiprocess=MULTIPROCESS, output_dir=out_dir, save_snapshots=SAVE_SNAPSHOTS)
    @info "Training complete" final_iterations=metrics_vec[end].iteration out_dir
end

# Keep interactive guard only. Do NOT call main() unconditionally so the file can be included
# (e.g. via Base.include(IncWrap, path)) without executing the training routine.
isinteractive() || main()
