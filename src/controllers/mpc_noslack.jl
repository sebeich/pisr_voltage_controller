using DataFrames
using Serialization
using Statistics
using LinearAlgebra
using Printf
using Plots
using BenchmarkTools
using MLJ
using CSV
using SymbolicRegression  # ensure operator symbols/types available
# (Removed JuMP/Ipopt; switching to global optimization for PQ tuning)
using BlackBoxOptim

# ---------------- Configuration ----------------
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const TEST_COMPLEX_CSV  = get(ENV, "TEST_COMPLEX_CSV", joinpath(REPO_ROOT, "data", "offline", "test_data_complex_blockrand.csv"))
const DEFAULT_MODEL_PATH = get(ENV, "MODEL_PATH", joinpath(REPO_ROOT, "models", "offline_highvar", "final_model_iter1020.jls"))

SLACK_BUS_ID = try parse(Int, get(ENV, "SLACK_BUS_ID", "66")) catch; 66 end
MODEL_PATH = get(ENV, "MODEL_PATH", DEFAULT_MODEL_PATH)
BENCH_ITERS = try parse(Int, get(ENV, "SR_BENCH_ITERS", "10")) catch; 10 end
# Verbosity toggle
const VERBOSE = lowercase(get(ENV, "VERBOSE", "false")) in ("1", "true", "yes", "y")
# Global optimization knobs (per-sample)
GLOB_MAXTIME   = try parse(Float64, get(ENV, "GLOB_MAXTIME", "0.1")) catch; 0.1 end   # ~10ms
GLOB_MAXEVALS  = try parse(Int,    get(ENV, "GLOB_MAXEVALS", "200")) catch; 200 end     # keep tiny
GLOB_POP       = try parse(Int,    get(ENV, "GLOB_POP", "20")) catch; 20 end            # small population
GLOB_METHOD    = Symbol(get(ENV, "GLOB_METHOD", "de_rand_1_bin_radiuslimited"))

# Control/penalty defaults (env overridable)
const COMMON_LIMIT = try parse(Float64, get(ENV, "COMMON_V_LIMIT", "1.05")) catch; 1.05 end
const QLIMIT       = try parse(Float64, get(ENV, "QLIMIT", "0.1")) catch; 0.1 end
const PLIMIT       = try parse(Float64, get(ENV, "PLIMIT", "0.1")) catch; 0.1 end
const P_WEIGHT     = try parse(Float64, get(ENV, "P_WEIGHT", "30.0")) catch; 30.0 end
const PENALTY      = try parse(Float64, get(ENV, "PENALTY_W", "1.0e4")) catch; 1.0e4 end

@assert isfile(TEST_COMPLEX_CSV) "Missing test complex CSV: $(TEST_COMPLEX_CSV)"
@assert isfile(MODEL_PATH) "Missing serialized model: $(MODEL_PATH)"

println("Evaluating model: $(MODEL_PATH)")
println("Using test data: $(TEST_COMPLEX_CSV)")
println("Slack bus: $(SLACK_BUS_ID)")

# ---------------- Robust Complex CSV Loader (shared) ----------------
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
    for (j, h) in enumerate(header)
        col_data = Vector{ComplexF64}(undef, n_samples)
        for i in 1:n_samples
            col_data[i] = parse_py_complex(raw_rows[i][j])
        end
        df[!, h] = col_data
    end
    return df
end

test_df = load_complex_df(TEST_COMPLEX_CSV)#[1:25, :]
println(@sprintf("Loaded test samples: %d", nrow(test_df)))

# ---------------- Prepare Features and Targets ----------------
voltage_cols = [c for c in names(test_df) if occursin(r"^V\d+_complex$", String(c))]
power_cols   = [c for c in names(test_df) if occursin(r"^S\d+_complex$", String(c))]
@assert !isempty(voltage_cols) "No voltage columns found in test data"
@assert !isempty(power_cols)   "No power columns found in test data"

voltage_bus_ids = sort(parse.(Int, match.(Ref(r"^V(\d+)_complex$"), String.(voltage_cols)) .|> x -> x.captures[1]))
@assert SLACK_BUS_ID in voltage_bus_ids "Slack bus $(SLACK_BUS_ID) missing in test data"
slack_col = Symbol(@sprintf("V%d_complex", SLACK_BUS_ID))
target_voltage_cols = filter(!=(slack_col), Symbol.(voltage_cols))

input_syms = [Symbol.(power_cols);]
inputs_df = DataFrame()
for c in Symbol.(power_cols); inputs_df[!, c] = test_df[!, c]; end

targets_df = DataFrame()
for c in target_voltage_cols; targets_df[!, c] = test_df[!, c]; end
X = NamedTuple{Tuple(input_syms)}(Tuple(inputs_df[!, c] for c in input_syms))
y_true = NamedTuple{Tuple(target_voltage_cols)}(Tuple(targets_df[!, c] for c in target_voltage_cols))

# ---------------- Custom Operators (must be defined BEFORE deserialize) ----------------
# These must match exactly the ones used during training so that the deserialized
# expression trees can re-bind to the existing methods.
line_current(V_send::Complex, S::Complex) = conj(S / V_send)
voltage_drop(I::Complex, Z::Complex) = I * Z

# ---------------- Load Model ----------------
mach = deserialize(MODEL_PATH)
println("Model loaded (custom operators registered).")

# Fast SR evaluator: extract best expressions and preallocate feature buffer
if !isdefined(@__MODULE__, :SR_REPORT)
    const SR_REPORT = report(mach)
end
if !isdefined(@__MODULE__, :SR_FUNS)
    const SR_FUNS = [SR_REPORT.equations[i][SR_REPORT.best_idx[i]] for i in eachindex(SR_REPORT.best_idx)]
end
const N_FEATURES = length(input_syms)
if !isdefined(@__MODULE__, :_Xscratch)
    const _Xscratch = Vector{ComplexF64}(undef, N_FEATURES)
end
if !isdefined(@__MODULE__, :_Xcol)
    const _Xcol = reshape(_Xscratch, N_FEATURES, 1)  # features x 1-sample matrix
end

# Build a reusable 1-sample NamedTuple input (vectors of length 1) from a single-row DataFrame slice
@inline function build_one_sample_nt(row_input::DataFrame, ordered_syms::Vector{Symbol})
    vals = Vector{Vector{ComplexF64}}(undef, length(ordered_syms))
    @inbounds for (i,sym) in enumerate(ordered_syms)
        vals[i] = [ComplexF64(row_input[1, sym])]
    end
    return NamedTuple{Tuple(ordered_syms)}(Tuple(vals))
end

@inline function max_abs_from_pred(row_pred)
    max_abs = 0.0
    @inbounds for col in values(row_pred)
        v = col[1]
        if v isa Complex
            a = abs(v)
            if a > max_abs; max_abs = a; end
        end
    end
    return max_abs
end

# --- SR evaluation helpers ---

# Utility: convert any numeric to ComplexF64 (reals get 0.0 imaginary)
@inline function _to_complex(z)
    if z isa Complex
        return ComplexF64(real(z), imag(z))
    else
        return ComplexF64(float(z), 0.0)
    end
end

# Fill feature buffer `_Xcol` from a 1-sample NamedTuple input
@inline function _fill_features_from_nt!(one_X_nt)
    @inbounds for i in 1:N_FEATURES
        _Xscratch[i] = _to_complex(one_X_nt[input_syms[i]][1])
    end
    return _Xcol
end

# Predict magnitude for a single target index using SR trees
@inline function _predict_one_abs_idx!(idx::Int, one_X_nt)
    _fill_features_from_nt!(one_X_nt)
    y = eval_tree_array(SR_FUNS[idx], _Xcol)
    # eval_tree_array can return (values, validmask/flag); take first component if tuple
    yr = y isa Tuple ? y[1] : y
    y1 = yr isa AbstractArray ? yr[1] : yr
    return abs(_to_complex(y1))
end

# Predict max |V| across all targets (full scan each time for consistency)
@inline function _predict_full_max_abs!(one_X_nt)
    _fill_features_from_nt!(one_X_nt)
    best_abs = 0.0
    @inbounds for i in 1:length(SR_FUNS)
        yi = eval_tree_array(SR_FUNS[i], _Xcol)
        yir = yi isa Tuple ? yi[1] : yi
        v = yir isa AbstractArray ? yir[1] : yir
        a = abs(_to_complex(v))
    if a > best_abs; best_abs = a; end
    end
    return best_abs
end

# Single entry point used everywhere (full scan for stability)
    @inline function _max_abs_predict!(one_X_nt)
        return _predict_full_max_abs!(one_X_nt)
    end

# Prediction-based rerun: mutate one_X_nt and compute max |V| using SR
@inline function rerun_with_xy!(one_X_nt, S_target, x::Float64, y::Float64)
    s = S_target isa Symbol ? S_target : Symbol(S_target)
    @inbounds begin
        S = one_X_nt[s][1]::ComplexF64
        # Do not clamp P here; keep it as original + delta
        newP = real(S) + y
        one_X_nt[s][1] = ComplexF64(newP, x)
    end
    return _max_abs_predict!(one_X_nt)
end

######## Line-wise Prediction (user-required) ########
println("Running line-wise prediction over $(nrow(inputs_df)) samples ...")
# Tiny JIT warmup of the SR path to stabilize first-iteration timing.
try
    row_input_wu = inputs_df[1:1, :]
    one_X_nt_wu = build_one_sample_nt(row_input_wu, Vector{Symbol}(input_syms))
    _max_abs_predict!(one_X_nt_wu)
    rerun_with_xy!(one_X_nt_wu, :S57_complex, imag(row_input_wu.S57_complex[1]), 0.0)
catch err
    if VERBOSE
        @warn "Warmup failed" err
    end
end
pred_rows = Vector{DataFrame}()
last_success_z = nothing  # warm-start vector from last successful optimization (if any)
# Track chosen P/Q adjustments per sample for plotting
q57_hist = Float64[]; p57_hist = Float64[]
q55_hist = Float64[]; p55_hist = Float64[]
q64_hist = Float64[]; p64_hist = Float64[]
# Track the updated S values (per-sample) so we can export totals and updated inputs
updated_input_rows = Vector{DataFrame}()
const CONTROLLED_S_SYMS = Symbol[ :S57_complex, :S55_complex, :S64_complex ]
"""
Lightweight DataFrame mutation for final prediction output only (not used inside optimizer).
"""
function rerun_with_xy_df!(row_input, S_target, x, y)
    # Do not clamp P here; keep it as original + delta
    row_input[!, S_target] = [ComplexF64(real(row_input[!, S_target][1]) + y, x)]
    row_pred = predict(mach, row_input)        # table with 1 row
    row_df = DataFrame(row_pred)
    complex_cols = [c for c in names(row_df) if eltype(row_df[!, c]) <: Complex]
    max_abs = maximum(abs, (row_df[1, c] for c in complex_cols))
    return max_abs, row_pred
end

# Batched variant: apply multiple updates and predict once (top-level scope)
function rerun_with_xy_df_multi!(row_input, updates)
    for (S_target, x, y) in updates
        s = S_target isa Symbol ? S_target : Symbol(S_target)
        row_input[!, s] = [ComplexF64(real(row_input[!, s][1]) + y, x)]
    end
    row_pred = predict(mach, row_input)
    row_df = DataFrame(row_pred)
    complex_cols = [c for c in names(row_df) if eltype(row_df[!, c]) <: Complex]
    max_abs = maximum(abs, (row_df[1, c] for c in complex_cols))
    return max_abs, row_pred
end
const OPT_LOG = DataFrame(sample=Int[], opt_time_s=Float64[], n_evals=Int[])
const N_EVALS = Ref(0)
const TIMING_LABEL = "PISR surrogate"
for i in 1:nrow(inputs_df)
    # Use global warm-start state within this loop
    global last_success_z
    row_input = inputs_df[i:i, :]              # single-row DataFrame slice
    # Baseline prediction via MLJ (used for skip decision)
    row_pred_base = predict(mach, row_input)
    row_df_base = DataFrame(row_pred_base)
    complex_cols_base = [c for c in names(row_df_base) if eltype(row_df_base[!, c]) <: Complex]
    base_max = maximum(abs, (row_df_base[1, c] for c in complex_cols_base))
    if VERBOSE; println("[Sample ", i, "] base max|V| = ", base_max); end

    # ---- Minimalistic P/Q optimization for three loads with voltage constraint ----
    common_limit = COMMON_LIMIT
    qlimit = QLIMIT
    plimit = PLIMIT

    # If either no violation or no control allowed, keep baseline prediction and skip optimization
    if base_max <= common_limit || (qlimit == 0.0 && plimit == 0.0)
        push!(pred_rows, DataFrame(row_pred_base))
        push!(updated_input_rows, deepcopy(row_input))
        push!(q57_hist, 0.0); push!(p57_hist, 0.0)
        push!(q55_hist, 0.0); push!(p55_hist, 0.0)
        push!(q64_hist, 0.0); push!(p64_hist, 0.0)
        continue
    end

    # Build a reusable 1-sample NamedTuple view of this row for fast predictions
    one_X_nt = build_one_sample_nt(row_input, Vector{Symbol}(input_syms))
    # Symbols and original values for mutation
    S57_sym = Symbol("S57_complex"); S55_sym = Symbol("S55_complex"); S64_sym = Symbol("S64_complex")
    origS57 = one_X_nt[S57_sym][1]
    origS55 = one_X_nt[S55_sym][1]
    origS64 = one_X_nt[S64_sym][1]

    function setup_var(z)
        qc1, pc1, qc2, pc2, qc3, pc3 = z
        # set candidate values directly on fast input (no P clamp)
        one_X_nt[S57_sym][1] = ComplexF64(real(origS57) + pc1, imag(origS57) + qc1)
        one_X_nt[S55_sym][1] = ComplexF64(real(origS55) + pc2, imag(origS55) + qc2)
        one_X_nt[S64_sym][1] = ComplexF64(real(origS64) + pc3, imag(origS64) + qc3)
        return one_X_nt, qc1, pc1, qc2, pc2, qc3, pc3
    end
    # evaluate max voltage magnitude via prediction path
    # Objective without internal clamping; enforce bounds via optimizer
    # Predict once per evaluation; mutate and restore in place.
    function objective_vec(z)
        N_EVALS[] += 1
        one_X_nt,qc1, pc1, qc2, pc2, qc3, pc3 = setup_var(z)
        max_abs_new = _max_abs_predict!(one_X_nt)
        # compute cost
        cost_p = P_WEIGHT * (pc1^2 + pc2^2 + pc3^2)
        cost_q = (qc1^2 + qc2^2 + qc3^2)
        vpen = max(0.0, max_abs_new - common_limit)
        return cost_p + cost_q + PENALTY * vpen^2
    end

    # Global derivative-free optimization with box constraints
    search_ranges = [(-qlimit, qlimit), (0.0, plimit),
                     (-qlimit, qlimit), (0.0, plimit),
                     (-qlimit, qlimit), (0.0, plimit)]
    # Warm-start with last successful candidate if available (clamped to bounds)
    function clamp_to_ranges(z)
        return Float64[
            clamp(z[1], -qlimit, qlimit), clamp(z[2], 0.0, plimit),
            clamp(z[3], -qlimit, qlimit), clamp(z[4], 0.0, plimit),
            clamp(z[5], -qlimit, qlimit), clamp(z[6], 0.0, plimit),
        ]
    end

    # Run optimizer 10 times, print timing for each, keep best
    best_f = Inf
    best_z = nothing
    N_EVALS[] = 0
    t = @elapsed begin
        if last_success_z === nothing
                # Ensure zero-deviation candidate is present
                res_run = bboptimize(objective_vec, zeros(6);
                    SearchRange = search_ranges,
                    NumDimensions = 6,
                    Method = GLOB_METHOD,
                    PopulationSize = GLOB_POP,
                    MaxFuncEvals = GLOB_MAXEVALS,
                    MaxTime = GLOB_MAXTIME,
                    TraceMode = :silent,
                    Seed = 42,
                )
        else
                init = clamp_to_ranges(last_success_z)
                res_run = bboptimize(objective_vec, init;
                    SearchRange = search_ranges,
                    NumDimensions = 6,
                    Method = GLOB_METHOD,
                    PopulationSize = GLOB_POP,
                    MaxFuncEvals = GLOB_MAXEVALS,
                    MaxTime = GLOB_MAXTIME,
                    TraceMode = :silent,
                    Seed = 42,
                )

        end
        # Evaluate result
        f = best_fitness(res_run)
        if f < best_f
                best_f = f
                best_z = best_candidate(res_run)
        end
    if VERBOSE; @printf("Sample %d optimization run %2d: %.2f ms (best_fitness=%.4e)\n", i, run_idx, t*1e3, best_f); end
    end
    push!(OPT_LOG, (i, t, N_EVALS[]))
    zopt = best_z === nothing ? zeros(6) : best_z

    # Extract deltas (keep P as delta; Q will be turned into absolute below)
    q57δ = clamp(zopt[1], -qlimit, qlimit); p57δ = clamp(zopt[2], 0.0, plimit)
    q55δ = clamp(zopt[3], -qlimit, qlimit); p55δ = clamp(zopt[4], 0.0, plimit)
    q64δ = clamp(zopt[5], -qlimit, qlimit); p64δ = clamp(zopt[6], 0.0, plimit)

    # If effectively no actions, do not rewrite S (avoid differences due to clamping)
    if isapprox(q57δ, 0.0; atol=1e-12) && isapprox(p57δ, 0.0; atol=1e-12) &&
       isapprox(q55δ, 0.0; atol=1e-12) && isapprox(p55δ, 0.0; atol=1e-12) &&
       isapprox(q64δ, 0.0; atol=1e-12) && isapprox(p64δ, 0.0; atol=1e-12)
        row_pred_final = predict(mach, row_input)
        push!(pred_rows, DataFrame(row_pred_final))
        push!(updated_input_rows, deepcopy(row_input))
        push!(q57_hist, 0.0); push!(p57_hist, 0.0)
        push!(q55_hist, 0.0); push!(p55_hist, 0.0)
        push!(q64_hist, 0.0); push!(p64_hist, 0.0)
        continue
    end

    # Build absolute Q and adjusted P (note: clamp applies to absolute P, not delta)
    q57_abs = imag(origS57) + q57δ
    q55_abs = imag(origS55) + q55δ
    q64_abs = imag(origS64) + q64δ

    p57_abs = real(origS57) + p57δ
    p55_abs = real(origS55) + p55δ
    p64_abs = real(origS64) + p64δ

    if VERBOSE; println("Optim PQ (57,55,64) -> ", (q57δ,p57δ,q55δ,p55δ,q64δ,p64δ), "  f=", best_f); end

    # Apply to fast input to check final max_abs (use abs Q and P = orig + delta)
    one_X_nt[S57_sym][1] = ComplexF64(p57_abs, q57_abs)
    one_X_nt[S55_sym][1] = ComplexF64(p55_abs, q55_abs)
    one_X_nt[S64_sym][1] = ComplexF64(p64_abs, q64_abs)
    max_abs = _max_abs_predict!(one_X_nt)
    if max_abs <= common_limit
        global last_success_z = copy(zopt)
    end

    # Update row_input for final logging in a single call (x=abs Q, y=P delta)
    _, row_pred = rerun_with_xy_df_multi!(row_input,
        [
            (:S57_complex, q57_abs, p57δ),
            (:S55_complex, q55_abs, p55δ),
            (:S64_complex, q64_abs, p64δ),
        ],
    )
    push!(pred_rows, DataFrame(row_pred))
    push!(updated_input_rows, deepcopy(row_input))

    # Store deltas (true adjustments)
    push!(q57_hist, q57δ); push!(p57_hist, p57δ)
    push!(q55_hist, q55δ); push!(p55_hist, p55δ)
    push!(q64_hist, q64δ); push!(p64_hist, p64δ)
end

# ---------------- Timing summary (Section V-C speed comparison) ----------------
CSV.write("optimization_timing.csv", OPT_LOG)
if nrow(OPT_LOG) > 0
    @printf("[timing] %d optimized samples: median optimization time %.2f ms, median %.0f objective evaluations, %.2f us per evaluation (%s)\n",
        nrow(OPT_LOG), 1e3 * median(OPT_LOG.opt_time_s), median(OPT_LOG.n_evals),
        1e6 * median(OPT_LOG.opt_time_s ./ max.(OPT_LOG.n_evals, 1)), TIMING_LABEL)
end

# Single-input inference time of the low-level predictor (all target buses, one input vector)
let nt = build_one_sample_nt(inputs_df[1:1, :], Vector{Symbol}(input_syms))
    t_inf = @belapsed _max_abs_predict!($nt)
    @printf("[timing] PISR inference (%d target equations, one input vector): %.2f us\n", length(SR_FUNS), 1e6 * t_inf)
end

predictions_df = vcat(pred_rows...)            # concatenate all single-row predictions

# Normalize column names to Symbols if they came back as Strings
if any(c -> c isa String, names(predictions_df))
    rename!(predictions_df, Dict(c => Symbol(c) for c in names(predictions_df)))
end
CSV.write("predictions.csv", predictions_df)
println("Saved predictions.csv with $(nrow(predictions_df)) rows and $(ncol(predictions_df)) columns (line-wise mode).")

# ---- Export updated node powers (complex S), incl. controlled-only view ----
if !isempty(updated_input_rows)
    # Build DataFrame of the updated inputs (one row per sample)
    updated_inputs_df = vcat(updated_input_rows...)
    # Keep only the S*_complex columns in original order
    s_syms = Symbol.(power_cols)
    s_only_df = updated_inputs_df[:, s_syms]

    # Write a CSV mirroring the input S columns, but with controlled nodes updated
    # Use Python-style complex tokens "(re+imj)" for compatibility with the loader
    function _cx_to_pystr(z::ComplexF64)
        return @sprintf("(%0.16f%+0.16fj)", real(z), imag(z))
    end

    # Convert updated S columns to python-style complex strings
    s_only_str_df = DataFrame()
    for c in s_syms
        s_only_str_df[!, c] = [_cx_to_pystr(ComplexF64(v)) for v in s_only_df[!, c]]
    end

    # Also convert predicted complex voltages and append to the same CSV (keep filename unchanged)
    v_pred_syms = [c for c in names(predictions_df) if occursin(r"^V\d+_complex$", String(c))]
    v_pred_syms = sort(v_pred_syms, by = x -> parse(Int, match(r"^V(\d+)_complex$", String(x)).captures[1]))
    v_only_str_df = DataFrame()
    for c in v_pred_syms
        v_only_str_df[!, c] = [_cx_to_pystr(ComplexF64(v)) for v in predictions_df[!, c]]
    end

    # Concatenate: first all updated S columns, then predicted V columns
    updated_with_v_df = hcat(s_only_str_df, v_only_str_df)
    CSV.write("updated_powers_complex.csv", updated_with_v_df)
    println("Saved updated_powers_complex.csv with updated S columns and predicted V columns.")

    # Also write a CSV with the previous controlled filename but now including ALL S columns
    # (unchanged for non-controlled nodes) plus predicted voltages, as requested.
    CSV.write("updated_powers_controlled_complex.csv", updated_with_v_df)
    println("Saved updated_powers_controlled_complex.csv with all S columns (controlled + unchanged) and predicted V columns.")
else
    println("No updated input rows captured; skipping export of updated powers and totals.")
end

# ---- Select bus to visualize ----
const PLOT_BUS = 57  # change if needed
target_sym = Symbol(@sprintf("V%d_complex", PLOT_BUS))
pred_col_syms = names(predictions_df)

has_target = target_sym in pred_col_syms || String(target_sym) in String.(pred_col_syms)
if !has_target
    println("Requested bus $(PLOT_BUS) not in predicted columns. Available example: ", first(pred_col_syms, min(10, length(pred_col_syms))))
    println("All predicted columns: ", pred_col_syms)
    println("(Debug) target_sym=", target_sym, " type=", typeof(target_sym))
else
    # Resolve actual column symbol used
    if !(target_sym in pred_col_syms)
        # find matching string then convert
        idx = findfirst(==(String(target_sym)), String.(pred_col_syms))
        target_sym = pred_col_syms[idx]
    end
    # Ground truth availability check
    has_truth = target_sym in names(targets_df)
    y_pred = predictions_df[!, target_sym]
    # Ensure vector of ComplexF64
    y_pred = ComplexF64.(y_pred)
    if has_truth
        y_true_vec = ComplexF64.(targets_df[!, target_sym])
        @assert length(y_true_vec) == length(y_pred)
    end

    # Plot magnitudes for ALL predicted bus voltages (not just target bus)
    voltage_pred_cols = [c for c in names(predictions_df) if occursin(r"^V\d+_complex$", String(c))]
    p2 = plot(title="All Bus |V| Magnitudes", xlabel="Sample", ylabel="|V| (p.u.)")
    for c in sort(voltage_pred_cols, by = x -> parse(Int, match(r"^V(\d+)_complex$", String(x)).captures[1]))
        plot!(p2, abs.(ComplexF64.(predictions_df[!, c])), label=String(c))
    end
    # Overlay true voltages if available
    if has_truth; plot!(p2, abs.(y_true_vec), label="True |V|", ls=:dash); end

    # Plot chosen adjustments (x=Q, y=P) for the three loads
    p3 = plot(title="P/Q Adjustments (x=Q, y=P)", xlabel="Sample", ylabel="Adjustment (p.u.)")
    plot!(p3, q57_hist, label="Q57 (x)")
    plot!(p3, p57_hist, label="P57 (y)")
    plot!(p3, q55_hist, label="Q55 (x)")
    plot!(p3, p55_hist, label="P55 (y)")
    plot!(p3, q64_hist, label="Q64 (x)")
    plot!(p3, p64_hist, label="P64 (y)")

    png_name = @sprintf("prediction_bus%d_plots.svg", PLOT_BUS)
    savefig(plot(p2, p3, layout=(2,1), size=(1000,1100)), png_name)
    println("Saved combined plot: $(png_name)")
end

