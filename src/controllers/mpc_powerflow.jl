using DataFrames
using Serialization
using Statistics
using LinearAlgebra
using Printf
using Plots
using BenchmarkTools
using MLJ
using CSV
using HTTP
using JSON3
# (Removed JuMP/Ipopt; switching to global optimization for PQ tuning)
using BlackBoxOptim

# ---------------- Configuration ----------------
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const TEST_COMPLEX_CSV  = get(ENV, "TEST_COMPLEX_CSV", joinpath(REPO_ROOT, "data", "offline", "test_data_complex_blockrand.csv"))

SLACK_BUS_ID = try parse(Int, get(ENV, "SLACK_BUS_ID", "66")) catch; 66 end
BENCH_ITERS = try parse(Int, get(ENV, "SR_BENCH_ITERS", "10")) catch; 10 end
# Verbosity toggle
const VERBOSE = lowercase(get(ENV, "VERBOSE", "false")) in ("1", "true", "yes", "y")
# Global optimization knobs (per-sample)
GLOB_MAXTIME   = try parse(Float64, get(ENV, "GLOB_MAXTIME", "20.0")) catch; 20.0 end   # ~10ms
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

# ------------------ External solver bridge ------------------
const SOLVER_URL = get(ENV, "SOLVER_URL", "http://127.0.0.1:8001/solve")

function _build_loads_from_row(row_input::DataFrame, power_syms::Vector{Symbol})
    loads = Vector{Dict{String,Any}}()
    for c in power_syms
        s = row_input[1, c]
        # ensure ComplexF64
        s_c = s isa Complex ? ComplexF64(s) : ComplexF64(float(s), 0.0)
        ext = parse(Int, String(c) |> x -> match(r"^S(\d+)_complex$", x).captures[1])
        push!(loads, Dict("bus_ext" => ext, "p_mw" => real(s_c), "q_mvar" => imag(s_c)))
    end
    return loads
end

function _call_solver_for_row(row_input::DataFrame; power_syms::Vector{Symbol}, updates=Vector{Tuple{Symbol,Float64,Float64}}())
    # Build base loads from all S*_complex columns
    loads = _build_loads_from_row(row_input, power_syms)
    # Apply updates: updates are tuples (S_sym, q_abs, p_abs) or (S_sym, q_abs, p_delta?)
    if !isempty(updates)
        # map ext->index
        idx_map = Dict{Int, Int}()
        for (i, ld) in enumerate(loads)
            idx_map[Int(ld["bus_ext"])] = i
        end
        for (S_sym, q_val, p_val) in updates
            sname = String(S_sym)
            m = match(r"S(\d+)_complex", sname)
            if m === nothing
                continue
            end
            ext = parse(Int, m.captures[1])
            if haskey(idx_map, ext)
                i = idx_map[ext]
                loads[i]["p_mw"] = float(p_val)
                loads[i]["q_mvar"] = float(q_val)
            else
                # create new load entry if missing
                push!(loads, Dict("bus_ext"=>ext, "p_mw"=>float(p_val), "q_mvar"=>float(q_val)))
            end
        end
    end

    body = Dict("loads" => loads)
    hdr = ["Content-Type" => "application/json"]
    resp = HTTP.post(SOLVER_URL, hdr, JSON3.write(body))
    if resp.status != 200
        error("Solver call failed with status=$(resp.status): $(String(resp.body))")
    end
    j = JSON3.read(String(resp.body))
    volt_map = get(j, "voltages", Dict())
    # Build row_pred-like Dict of Symbol => Vector{ComplexF64}
    row_pred = Dict{Symbol, Vector{ComplexF64}}()
    max_abs = 0.0
    for (k, v) in volt_map
        # k like "V57_complex" and v is string "+0.999...+0.001...j"
        sym = Symbol(k)
        cz = try
            parse_py_complex(String(v))
        catch
            ComplexF64(0.0, 0.0)
        end
        row_pred[sym] = [cz]
        a = abs(cz)
        if a > max_abs; max_abs = a; end
    end
    return max_abs, row_pred
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

# Removed SymbolicRegression-based prediction. This script uses the external
# powerflow API for all per-iteration voltage predictions.

######## Line-wise Prediction (user-required) ########
println("Running line-wise prediction over $(nrow(inputs_df)) samples ...")
pred_rows = Vector{DataFrame}()
# Track chosen P/Q adjustments per sample for plotting
q57_hist = Float64[]; p57_hist = Float64[]
q55_hist = Float64[]; p55_hist = Float64[]
q64_hist = Float64[]; p64_hist = Float64[]
# Track the updated S values (per-sample) so we can export totals and updated inputs
updated_input_rows = Vector{DataFrame}()
const CONTROLLED_S_SYMS = Symbol[ :S57_complex, :S55_complex, :S64_complex ]

const OPT_LOG = DataFrame(sample=Int[], opt_time_s=Float64[], n_evals=Int[])
const N_EVALS = Ref(0)
const TIMING_LABEL = "pandapower runpp via solver API"
for i in 1:nrow(inputs_df)
    row_input = inputs_df[i:i, :]
    # Baseline prediction via external solver
    power_syms = Symbol.(power_cols)
    base_max, row_pred_base = _call_solver_for_row(row_input; power_syms=power_syms)
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

    # Read original S values directly from the DataFrame row
    origS57 = ComplexF64(row_input[1, :S57_complex])
    origS55 = ComplexF64(row_input[1, :S55_complex])
    origS64 = ComplexF64(row_input[1, :S64_complex])

    # Objective calls the external solver for each candidate.
    function objective_vec(z)
        N_EVALS[] += 1
        qc1, pc1, qc2, pc2, qc3, pc3 = z
        # absolute values for Q and P
        q57_abs = imag(origS57) + qc1
        p57_abs = real(origS57) + pc1
        q55_abs = imag(origS55) + qc2
        p55_abs = real(origS55) + pc2
        q64_abs = imag(origS64) + qc3
        p64_abs = real(origS64) + pc3
        updates = [(:S57_complex, q57_abs, p57_abs), (:S55_complex, q55_abs, p55_abs), (:S64_complex, q64_abs, p64_abs)]
        max_abs_new, _ = _call_solver_for_row(row_input; power_syms=Symbol.(power_cols), updates=updates)
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

    # Run optimizer
    N_EVALS[] = 0
    t_opt = @elapsed res_run = bboptimize(objective_vec, zeros(6);
        SearchRange = search_ranges,
        NumDimensions = 6,
        Method = GLOB_METHOD,
        PopulationSize = GLOB_POP,
        MaxFuncEvals = GLOB_MAXEVALS,
        MaxTime = GLOB_MAXTIME,
        TraceMode = :silent,
        Seed = 42,
    )
    push!(OPT_LOG, (i, t_opt, N_EVALS[]))
    best_f = best_fitness(res_run)
    best_z = best_candidate(res_run)
    zopt = best_z === nothing ? zeros(6) : best_z

    # Extract deltas (keep P as delta; Q will be turned into absolute below)
    q57δ = clamp(zopt[1], -qlimit, qlimit); p57δ = clamp(zopt[2], 0.0, plimit)
    q55δ = clamp(zopt[3], -qlimit, qlimit); p55δ = clamp(zopt[4], 0.0, plimit)
    q64δ = clamp(zopt[5], -qlimit, qlimit); p64δ = clamp(zopt[6], 0.0, plimit)

    # If effectively no actions, do not rewrite S (avoid differences due to clamping)
    if isapprox(q57δ, 0.0; atol=1e-12) && isapprox(p57δ, 0.0; atol=1e-12) &&
       isapprox(q55δ, 0.0; atol=1e-12) && isapprox(p55δ, 0.0; atol=1e-12) &&
       isapprox(q64δ, 0.0; atol=1e-12) && isapprox(p64δ, 0.0; atol=1e-12)
        # No changes: use solver baseline result
        _, row_pred_final = _call_solver_for_row(row_input; power_syms=Symbol.(power_cols))
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

    # Apply update via external solver to check final max_abs
    max_abs, _ = _call_solver_for_row(row_input; power_syms=Symbol.(power_cols), updates=[(:S57_complex, q57_abs, p57_abs), (:S55_complex, q55_abs, p55_abs), (:S64_complex, q64_abs, p64_abs)])

    # Update row_input for final logging
    @time _, row_pred = _call_solver_for_row(row_input; power_syms=Symbol.(power_cols), updates=[(:S57_complex, q57_abs, p57_abs), (:S55_complex, q55_abs, p55_abs), (:S64_complex, q64_abs, p64_abs)])
    push!(pred_rows, DataFrame(row_pred))
    # Apply the updated S values into the saved input row for export
    updated_row = deepcopy(row_input)
    updated_row[1, :S57_complex] = ComplexF64(p57_abs, q57_abs)
    updated_row[1, :S55_complex] = ComplexF64(p55_abs, q55_abs)
    updated_row[1, :S64_complex] = ComplexF64(p64_abs, q64_abs)
    push!(updated_input_rows, updated_row)

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
    CSV.write("updated_powers_complex_powerflow.csv", updated_with_v_df)
    println("Saved updated_powers_complex_powerflow.csv with updated S columns and predicted V columns.")

    # Also write a CSV with the previous controlled filename but now including ALL S columns
    # (unchanged for non-controlled nodes) plus predicted voltages, as requested.
    CSV.write("updated_powers_controlled_complex_powerflow.csv", updated_with_v_df)
    println("Saved updated_powers_controlled_complex_powerflow.csv with all S columns (controlled + unchanged) and predicted V columns.")
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

    png_name = @sprintf("prediction_bus%d_plots_powerflow.svg", PLOT_BUS)
    savefig(plot(p2, p3, layout=(2,1), size=(1000,1100)), png_name)
    println("Saved combined plot: $(png_name)")
end

