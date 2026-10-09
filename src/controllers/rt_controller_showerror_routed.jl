#!/usr/bin/env -S julia --project=@.

# Real-time SR-based MPC controller for the pandapower service.
# - Polls the powerflow service /state every 100ms to fetch current S (MW/MVAr)
# - Builds a one-row input matching the SR model features (from a baseline CSV)
# - Runs a tiny black-box optimization (<= 100ms) to choose ΔP, ΔQ for selected buses
# - Posts the computed deltas (MW/MVAr) back via /mods
#
# Notes
# - Feature units: use MW/MVAr directly (service and datasets store MW/MVAr tokens).
# - Controlled buses default to [57, 55, 64] but will auto-intersect with /status output.
# - Optimization mirrors logic in mpc_noslack.jl (penalty on |V|max beyond limit, small ΔP/ΔQ costs).

using DataFrames
using Serialization
using Statistics
using LinearAlgebra
using Printf
using Dates
using MLJ
using SymbolicRegression
using BlackBoxOptim
using HTTP
using JSON3

# ---------------- Configuration ----------------
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_MODEL_PATH = get(ENV, "MODEL_PATH", joinpath(REPO_ROOT, "models", "cross_eval", "runs", "run_09", "training", "final_model_iter1020.jls"))

const PF_BASE_URL = get(ENV, "PF_BASE_URL", "http://127.0.0.1:8000")
const CONTROLLER_SOURCE = "pisr"
const LOOP_PERIOD_S = try parse(Float64, get(ENV, "LOOP_PERIOD_S", "0.1")) catch; 0.1 end
const S_BASE_MVA = try parse(Float64, get(ENV, "S_BASE_MVA", "0.2")) catch; 0.2 end
const SLACK_BUS_ID = try parse(Int, get(ENV, "SLACK_BUS_ID", "66")) catch; 66 end

# Global optimization knobs (per-iteration)
const GLOB_MAXTIME = try parse(Float64, get(ENV, "GLOB_MAXTIME", "0.1")) catch; 0.1 end   # seconds
const GLOB_MAXEVALS = try parse(Int, get(ENV, "GLOB_MAXEVALS", "500")) catch; 500 end
const GLOB_POP = try parse(Int, get(ENV, "GLOB_POP", "20")) catch; 20 end
const GLOB_METHOD = Symbol(get(ENV, "GLOB_METHOD", "de_rand_1_bin_radiuslimited"))

# Control/penalty defaults (env overridable)
const COMMON_LIMIT = try parse(Float64, get(ENV, "COMMON_V_LIMIT", get(ENV, "VMAX", "1.05"))) catch; 1.05 end
const QLIMIT = try parse(Float64, get(ENV, "QLIMIT", "0.1")) catch; 0.1 end
const PLIMIT = try parse(Float64, get(ENV, "PLIMIT", "0.3")) catch; 0.3 end
const P_WEIGHT = try parse(Float64, get(ENV, "P_WEIGHT", "30.0")) catch; 30.0 end
const Q_WEIGHT = try parse(Float64, get(ENV, "Q_WEIGHT", "1.0")) catch; 1.0 end
const PENALTY = try parse(Float64, get(ENV, "PENALTY_W", "1.0e4")) catch; 1.0e4 end
const VERBOSE = lowercase(get(ENV, "VERBOSE", "false")) in ("1", "true", "yes", "y")
const DEBUG_SR_INPUTS = lowercase(get(ENV, "DEBUG_SR_INPUTS", "false")) in ("1", "true", "yes", "y")

# Default controlled buses; can override via env CONTROLLED_BUSES="55,57,64"
const CONTROLLED_BUSES = let s = strip(get(ENV, "CONTROLLED_BUSES", "55,57,64"))
    try
        parts = split(s, [',', ' ']; keepempty=false)
        Int[parse(Int, p) for p in parts if !isempty(p)]
    catch
        [55, 57, 64]
    end
end

@assert isfile(DEFAULT_MODEL_PATH) "Missing serialized model: $(DEFAULT_MODEL_PATH)"

if DEBUG_SR_INPUTS
    println("[rtctl] Model: $(DEFAULT_MODEL_PATH)")
    println(@sprintf("[rtctl] Loop %.0f ms, max opt time %.0f ms", LOOP_PERIOD_S*1e3, GLOB_MAXTIME*1e3))
end

# ---------------- Compat: module for custom ops used during training ----------------
# The serialized SR model may reference IncWrap.line_current / IncWrap.voltage_drop
# (when training was run via collect_and_train_from_api.jl which wraps code in IncWrap).
# If missing, include a small module definition file.
if !isdefined(Main, :IncWrap)
    incwrap_path = joinpath(REPO_ROOT, "src", "pisr", "incwrap_ops.jl")
    if isfile(incwrap_path)
        include(incwrap_path)
    else
        @warn "Missing incwrap_ops.jl; falling back to local operator defs"
        # Fallback: define the needed functions in Main to avoid resolution errors
        line_current(V_send::Complex, S::Complex) = conj(S / V_send)
        voltage_drop(I::Complex, Z::Complex) = I * Z
    end
end

"""
Build and maintain the input schema entirely from the API; no CSV dependency.
Initializes:
- input_syms: Vector{Symbol} of S{bus}_complex, sorted by bus id
- input_row_df: 1-row DataFrame with same columns, filled from /state power (totals used in last PF)
- N_FEATURES, _Xscratch, _Xcol: buffers for fast SR eval
- SR_TARGETS: symbols of predicted V{bus}_complex from the model (filtered to non-slack and buses with S)
- PRED_BUS_IDS: corresponding bus ids used for measured max filtering
- controlled: selected control buses intersected with available features and /status
"""
input_row_df = DataFrame()
input_syms = Symbol[]
N_FEATURES = 0
_Xscratch = ComplexF64[]
_Xcol = zeros(ComplexF64, 0, 1)
SR_TARGETS = Symbol[]
PRED_BUS_IDS = Int[]
controlled = Int[]

const TRAIN_INPUT_ORDER = Symbol[
    Symbol("S64_complex"),
    Symbol("S57_complex"),
    Symbol("S55_complex"),
    Symbol("S61_complex"),
]

const TRAIN_TARGET_ORDER = Symbol[
    Symbol("V64_complex"),
    Symbol("V57_complex"),
    Symbol("V55_complex"),
    Symbol("V61_complex"),
]

# Custom operators matching training time
line_current(V_send::Complex, S::Complex) = conj(S / V_send)
voltage_drop(I::Complex, Z::Complex) = I * Z

mach = deserialize(DEFAULT_MODEL_PATH)
if DEBUG_SR_INPUTS
    println("[rtctl] Model loaded.")
end

# Extract SR equations for fast evaluation
const SR_REPORT = report(mach)
const SR_FUNS = [SR_REPORT.equations[i][SR_REPORT.best_idx[i]] for i in eachindex(SR_REPORT.best_idx)]

@inline function build_one_sample_nt(row_input::DataFrame, ordered_syms::Vector{Symbol})
    vals = Vector{Vector{ComplexF64}}(undef, length(ordered_syms))
    @inbounds for (i,sym) in enumerate(ordered_syms)
        vals[i] = [ComplexF64(row_input[1, sym])]
    end
    return NamedTuple{Tuple(ordered_syms)}(Tuple(vals))
end

@inline _to_complex(z) = z isa Complex ? ComplexF64(real(z), imag(z)) : ComplexF64(float(z), 0.0)

@inline function _fill_features_from_nt!(one_X_nt)
    @inbounds for i in 1:N_FEATURES
        _Xscratch[i] = _to_complex(one_X_nt[input_syms[i]][1])
    end
    return _Xcol
end

@inline function _predict_full_max_abs!(one_X_nt)
    _fill_features_from_nt!(one_X_nt)
    best_abs = 0.0
    @inbounds for i in 1:length(SR_FUNS)
        yi = eval_tree_array(SR_FUNS[i], _Xcol)
        yir = yi isa Tuple ? yi[1] : yi
        v = yir isa AbstractArray ? yir[1] : yir
        a = abs(_to_complex(v))
        if a > best_abs
            best_abs = a
        end
    end
    return best_abs
end

@inline _max_abs_predict!(one_X_nt) = _predict_full_max_abs!(one_X_nt)

@inline function _predict_namedtuple(one_X_nt)
    # Use MLJ to get a NamedTuple of predictions keyed by symbols (matches training target names)
    return predict(mach, one_X_nt)
end

# Robust extractor for MLJ prediction entries -> scalar magnitude (Float64)
@inline function get_pred_mag(val)
    # val can be a scalar, Complex, or a 1-element array/tuple containing a Complex
    v = val
    if v isa AbstractArray || v isa Tuple
        isempty(v) && return NaN
        v = v[1]
    end
    if v === nothing
        return NaN
    elseif v isa Complex
        return abs(v)
    elseif v isa Number
        return float(v)
    else
        try
            return abs(parse(ComplexF64, String(v)))
        catch
            return NaN
        end
    end
end

function rerun_with_xy!(one_X_nt, S_target, x::Float64, y::Float64)
    s = S_target isa Symbol ? S_target : Symbol(S_target)
    @inbounds begin
        S = one_X_nt[s][1]::ComplexF64
        newP = real(S) + y
        one_X_nt[s][1] = ComplexF64(newP, x)
    end
    return _max_abs_predict!(one_X_nt)
end

# Warmup JIT will be performed after schema initialization

# ---------------- HTTP helpers ----------------
function http_json(method::String, url::String; body=nothing, timeout_s::Float64=0.2)
    headers = Dict("Content-Type" => "application/json")
    rt = max(1, ceil(Int, timeout_s))
    last_err = nothing
    for attempt in 1:3
        try
            m = uppercase(method)
            r = if m == "GET"
                HTTP.get(url; readtimeout=rt)
            elseif m == "DELETE"
                HTTP.request("DELETE", url; readtimeout=rt)
            elseif m == "POST"
                data = body === nothing ? UInt8[] : JSON3.write(body)
                HTTP.post(url; headers=headers, body=data, readtimeout=rt)
            else
                data = body === nothing ? UInt8[] : JSON3.write(body)
                HTTP.request(m, url, headers, data; readtimeout=rt)
            end
            if r.status >= 200 && r.status < 300
                return JSON3.read(String(r.body))
            end
            error("HTTP $(method) $(url) failed with $(r.status)")
        catch err
            last_err = err
            attempt == 3 || sleep(0.05 * attempt)
        end
    end
    throw(last_err)
end

function parse_cx_json(s::AbstractString)
    # Expect like "+0.012345+0.067890j" or "-0.01-0.02j"
    ss = strip(String(s))
    # Fast path: let Julia parse Python-like by replacing 'j' with 'im'
    if occursin('j', ss)
        s2 = replace(ss, 'j' => "im")
        try
            return parse(ComplexF64, s2)
        catch
            # fallthrough to regex
        end
    end
    m = match(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([+-](?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)j$", ss)
    m === nothing && return 0.0 + 0.0im
    return ComplexF64(parse(Float64, m.captures[1]), parse(Float64, m.captures[2]))
end

@inline _fmt_c(z::Complex) = @sprintf("%.6f%+.6fj", real(z), imag(z))

function _get_pw_val(pw, key::String)
    v = nothing
    try
        v = get(pw, key, nothing)
    catch
        try
            v = haskey(pw, key) ? pw[key] : nothing
        catch
            v = nothing
        end
    end
    if v === nothing
        return ComplexF64(0,0)
    end
    try
        return parse_cx_json(String(v))
    catch
        return v isa Number ? ComplexF64(v, 0.0) : ComplexF64(0,0)
    end
end

function get_state()
    return http_json("GET", PF_BASE_URL * "/state"; timeout_s=0.2)
end

function get_status()
    return http_json("GET", PF_BASE_URL * "/status"; timeout_s=0.2)
end

function post_mods(items)
    body = (; items)
    return http_json("POST", PF_BASE_URL * "/controller/$(CONTROLLER_SOURCE)/mods"; body=body, timeout_s=0.2)
end

function is_inactive_controller_conflict(err)::Bool
    msg = sprint(showerror, err)
    return occursin("409", msg) && occursin("is not active", msg)
end

function clear_controller_mods()
    return http_json("DELETE", PF_BASE_URL * "/controller/$(CONTROLLER_SOURCE)/mods"; timeout_s=0.2)
end

function controller_enabled()
    js = http_json("GET", PF_BASE_URL * "/controller/$(CONTROLLER_SOURCE)/enabled"; timeout_s=0.2)
    return Bool(get(js, "enabled", false))
end

function get_current_routed_mods()
    st = get_state()
    return _extract_current_mods(st)
end

function post_cost(source::AbstractString, value; detail=Dict{String,Any}())
    body = (; source=String(source), value=value, detail=detail)
    return http_json("POST", PF_BASE_URL * "/cost"; body=body, timeout_s=0.2)
end

# ---------------- Schema initialization from API ----------------
function init_schema_from_api!()
    # 1) Discover features from /state power (fallback to power_orig)
    st = get_state()
    pw = haskey(st, "power") ? st["power"] : get(st, "power_orig", Dict{String,Any}())
    keys_s = String[String(k) for k in keys(pw) if occursin(r"^S\d+_complex$", String(k))]
    isempty(keys_s) && error("No S*_complex keys found in /state")
    global input_syms
    available_s = Set(Symbol.(keys_s))
    ordered_inputs = [sym for sym in TRAIN_INPUT_ORDER if sym in available_s]
    isempty(ordered_inputs) && error("None of the expected routed SR inputs were found in /state")
    input_syms = ordered_inputs

    # 2) Initialize the 1-row input DataFrame with current values
    empty!(input_row_df)
    for sym in input_syms
        input_row_df[!, sym] = [0.0 + 0.0im]
    end
    update_input_row_from_state!(input_row_df, st)

    # 3) Feature buffers for fast eval
    global N_FEATURES, _Xscratch, _Xcol
    N_FEATURES = length(input_syms)
    _Xscratch = Vector{ComplexF64}(undef, N_FEATURES)
    _Xcol = reshape(_Xscratch, N_FEATURES, 1)

    # 4) Determine SR targets by running a single predict and reading the keys
    one_nt = build_one_sample_nt(input_row_df, Vector{Symbol}(input_syms))
    println(one_nt)
    ypre = predict(mach, one_nt)
    println(ypre)
    ys = Set(Symbol.(collect(propertynames(ypre))))
    filtered = [sym for sym in TRAIN_TARGET_ORDER if sym in ys]
    global SR_TARGETS, PRED_BUS_IDS
    SR_TARGETS = filtered
    PRED_BUS_IDS = sort!(unique([parse(Int, match(r"^V(\d+)_complex$", String(s)).captures[1]) for s in SR_TARGETS]))

    # 5) Controlled buses: force to CONTROLLED_BUSES (env default 55,57,64); intersect with available S features
    preferred = CONTROLLED_BUSES
    sbus_list = sort!(Int[parse(Int, match(r"^S(\d+)_complex$", String(s)).captures[1]) for s in input_syms])
    global controlled
    controlled = [b for b in preferred if b in sbus_list]
    dropped = setdiff(preferred, controlled)
    if !isempty(dropped)
        @warn "Some requested control buses not present in SR features, dropping" dropped
    end
    if DEBUG_SR_INPUTS
        println("[rtctl] Using controlled buses: ", controlled)
    end
    # Diagnostics: show feature columns discovered
    try
        if DEBUG_SR_INPUTS
            println("[rtctl] SR feature columns (", length(input_syms), "): ", join(string.(input_syms), ", "))
        end
    catch
        # ignore
    end

    # 6) Warmup JIT
    try
        _max_abs_predict!(one_nt)
    catch err
        @warn "Warmup failed" err
    end
end

function init_schema_from_api_with_retries!(; timeout_s::Float64=180.0)
    deadline = time() + timeout_s
    last_err = nothing
    while time() < deadline
        try
            init_schema_from_api!()
            return nothing
        catch err
            last_err = err
            @warn "Schema initialization from API failed; retrying" err
            sleep(0.5)
        end
    end
    last_err === nothing || throw(last_err)
    error("Schema initialization from API failed without an exception")
end

# ---------------- Control set resolution ----------------
# Handled in init_schema_from_api! using /status and available S features

# ---------------- Main control loop ----------------
last_post = nothing
if DEBUG_SR_INPUTS
    println("[rtctl] Starting control loop...")
end

# Warm-start vector for optimizer (persist across iterations), mirrors mpc_noslack
last_success_z = nothing  # Vector{Float64} of length 2*length(controlled) when set
was_enabled = false

function optimize_once!(input_row_df::DataFrame; current_mods::Dict{Int,Tuple{Float64,Float64}}=Dict{Int,Tuple{Float64,Float64}}())
    # Build fast input
    one_X_nt = build_one_sample_nt(input_row_df, Vector{Symbol}(input_syms))
    # Baseline |V|max
    base_max = _max_abs_predict!(one_X_nt)

    # Collect symbols and original values
    syms = Symbol.(string.(:S, controlled, :_complex))
    origS = ComplexF64[ one_X_nt[s][1] for s in syms ]
    # Current mods aligned with syms order
    curP = Float64[]; curQ = Float64[]
    for bus in controlled
        cur = get(current_mods, bus, (0.0, 0.0))
        push!(curP, cur[1]); push!(curQ, cur[2])
    end
    base_vpen = max(0.0, base_max - COMMON_LIMIT)
    base_cost = P_WEIGHT * sum(abs2, curP) + Q_WEIGHT * sum(abs2, curQ) + PENALTY * base_vpen^2
    # If under limit and no current mods, we can skip
    if (base_max <= COMMON_LIMIT) && all(abs.(curP) .<= 1e-9) && all(abs.(curQ) .<= 1e-9)
        return Dict{Int,Tuple{Float64,Float64}}(), base_max, base_max, Dict{Int,Float64}(), base_cost
    end

    function setup_var(z)
        # z layout: [qc1, pc1, qc2, pc2, ...]
        for i in eachindex(syms)
            qc = z[2*i - 1]; pc = z[2*i]
            # Enforce total bounds for prediction: PV plants cannot produce negative P,
            # so clamp the total P to [0, PLIMIT]. Q remains symmetric within [-QLIMIT, QLIMIT].
            tot_p = clamp(curP[i] + pc, 0.0, PLIMIT)
            tot_q = clamp(curQ[i] + qc, -QLIMIT, QLIMIT)
            eff_pc = tot_p - curP[i]
            eff_qc = tot_q - curQ[i]
            one_X_nt[syms[i]][1] = ComplexF64(real(origS[i]) + eff_pc, imag(origS[i]) + eff_qc)
        end
        return one_X_nt
    end

    function objective_vec(z)
        setup_var(z)
        # Fast bound via direct tree eval across targets
        max_abs_new = _max_abs_predict!(one_X_nt)
        effective_max = max_abs_new
        # costs
        cost_p = 0.0
        cost_q = 0.0
        for i in 1:length(syms)
            pc = z[2*i]
            qc = z[2*i - 1]
            # Enforce total P non-negative for PV (clamp to [0, PLIMIT]) before cost
            total_p = clamp(curP[i] + pc, 0.0, PLIMIT)
            total_q = clamp(curQ[i] + qc, -QLIMIT, QLIMIT)
            cost_p += total_p^2
            cost_q += total_q^2
        end
    vpen = max(0.0, effective_max - COMMON_LIMIT)
        return P_WEIGHT * cost_p + Q_WEIGHT * cost_q + PENALTY * vpen^2
    end

    # bounds per var (allow reducing mods): symmetric around 0
    ranges = Vector{Tuple{Float64,Float64}}(undef, 2*length(syms))
    for i in 1:length(syms)
        ranges[2*i - 1] = (-QLIMIT, QLIMIT)  # qc delta
        ranges[2*i]     = (-PLIMIT, PLIMIT)  # pc delta
    end

    # Optional warm-start with last successful candidate (clamped and size-checked)
    function clamp_to_ranges!(z::Vector{Float64})
        @inbounds for i in 1:length(syms)
            z[2*i - 1] = clamp(z[2*i - 1], -QLIMIT, QLIMIT) # qc delta
            z[2*i]     = clamp(z[2*i], -PLIMIT, PLIMIT)     # pc delta
        end
        return z
    end

    init_z = if last_success_z isa Vector{Float64} && length(last_success_z) == 2*length(syms)
        clamp_to_ranges!(copy(last_success_z))
    else
        zeros(2*length(syms))
    end

    res = bboptimize(objective_vec, init_z;
        SearchRange = ranges,
        NumDimensions = length(ranges),
        Method = GLOB_METHOD,
        PopulationSize = GLOB_POP,
        MaxFuncEvals = GLOB_MAXEVALS,
        MaxTime = GLOB_MAXTIME,
        TraceMode = :silent,
        Seed = 42,
    )

    best_z = best_candidate(res)
    if best_z === nothing
        return Dict{Int,Tuple{Float64,Float64}}(), base_max, base_max, Dict{Int,Float64}(), base_cost
    end
    objective_best = try
        Float64(objective_vec(best_z))
    catch
        NaN
    end
    # Clamp and extract
    deltas = Dict{Int,Tuple{Float64,Float64}}()
    for (i, bus) in enumerate(controlled)
        qc = clamp(best_z[2*i - 1], -QLIMIT, QLIMIT)
        # Allow negative active-power changes so the optimizer can reduce previously
        # applied dP values back towards zero when the voltage permits. Previously
        # this was clamped to [0, PLIMIT] which prevented reducing active mods.
        pc = clamp(best_z[2*i], -PLIMIT, PLIMIT)
        deltas[bus] = (pc, qc)  # (ΔP_pu, ΔQ_pu)
    end
    # Compute predicted post-change max |V| and optionally update warm-start
    # Apply all deltas to one_X_nt
    # Apply deltas to compute post-change prediction, but ensure total P respects PV lower bound 0
    for (i, bus) in enumerate(controlled)
        pc, qc = deltas[bus]
        tot_p = clamp(real(origS[i]) + pc, 0.0, PLIMIT)
        tot_q = clamp(imag(origS[i]) + qc, -QLIMIT, QLIMIT)
        one_X_nt[syms[i]][1] = ComplexF64(tot_p, tot_q)
    end
    post_max = _max_abs_predict!(one_X_nt)
    # Also compute per-target predicted magnitudes for downstream reporting
    post_named = _predict_namedtuple(one_X_nt)
    post_bus_local = Dict{Int,Float64}()
    for sym in SR_TARGETS
        m = match(r"^V(\d+)_complex$", String(sym))
        m === nothing && continue
        bus = parse(Int, m.captures[1])
        post_bus_local[bus] = get_pred_mag(post_named[sym])
    end
    # Update warm-start only when within limit (mirrors mpc_noslack behavior)
    if post_max <= COMMON_LIMIT
        global last_success_z
        last_success_z = copy(best_z)
    end
    return deltas, base_max, post_max, post_bus_local, objective_best
end

function update_input_row_from_state!(input_row_df::DataFrame, st)
    # Baseline for SR is TOTAL S used in the last PF step: st["power"] (targets + current mods).
    # We ADD dp/dq during optimization on top of this baseline, meaning changes are relative to current mods.
    # Robust getters for JSON3.Object or Dict-like
    getobj(obj, key::String) = try
        haskey(obj, key) ? obj[key] : nothing
    catch
        try
            get(obj, key, nothing)
        catch
            nothing
        end
    end
    pw = getobj(st, "power")
    pw === nothing && (pw = getobj(st, "power_orig"))
    pw === nothing && return

    # Update all expected SR feature columns deterministically using robust getter
    local assigned = 0
    for sym in input_syms
        key = String(sym)
        z = _get_pw_val(pw, key)
        z_pwr = ComplexF64(real(z), imag(z))
        if !(sym in names(input_row_df))
            input_row_df[!, sym] = [z_pwr]
        else
            input_row_df[1, sym] = z_pwr
        end
        assigned += 1
    end
    # Optional focused debug for controlled buses
    if DEBUG_SR_INPUTS && assigned > 0
        io = IOBuffer()
        #print(io, "    updated inputs:")
        for bus in controlled
            sym = Symbol(@sprintf("S%d_complex", bus))
            val = sym in names(input_row_df) ? ComplexF64(input_row_df[1, sym]) : ComplexF64(0,0)
           # print(io, ' ', @sprintf("%s=%s", String(sym), _fmt_c(val)))
        end
        println(String(take!(io)))
    end
end

function _extract_current_mods(st)
    getobj(obj, key::String) = try
        haskey(obj, key) ? obj[key] : nothing
    catch
        try
            get(obj, key, nothing)
        catch
            nothing
        end
    end
    mods = Dict{Int,Tuple{Float64,Float64}}()
    try
        raw = getobj(st, "mods")
        raw === nothing && return mods
        for (k,v) in pairs(raw)
            bus = try parse(Int, String(k)) catch; Int(k) end
            dp = try Float64(getobj(v, "dP_mw")) catch; 0.0 end
            dq = try Float64(getobj(v, "dQ_mvar")) catch; 0.0 end
            mods[bus] = (dp, dq)
        end
    catch
        # ignore
    end
    return mods
end

function maybe_post_deltas!(deltas_mw::Dict{Int,Tuple{Float64,Float64}}, current_mods::Dict{Int,Tuple{Float64,Float64}})
    if isempty(deltas_mw)
        return nothing
    end
    # Compute absolute mods = current_mod + delta, then send
    items = Any[]
    for (bus, (dp_mw, dq_mvar)) in deltas_mw
        cur = get(current_mods, bus, (0.0, 0.0))
    # Ensure total P (cur + dp) is non-negative for PV plants
    new_dp = clamp(cur[1] + dp_mw, 0.0, PLIMIT)
        new_dq = clamp(cur[2] + dq_mvar, -QLIMIT, QLIMIT)
        push!(items, (; bus_ext=bus, dP_mw=new_dp, dQ_mvar=new_dq))
    end
    try
        resp = post_mods(items)
        if VERBOSE && DEBUG_SR_INPUTS
            println("[rtctl] POST /mods -> ", resp)
        end
        return resp
    catch err
        if is_inactive_controller_conflict(err)
            return nothing
        end
        @warn "Failed posting mods" err
        return nothing
    end
end

function maybe_post_cost!(cost_value; base_max::Float64=NaN, post_max::Float64=NaN, meas_max::Float64=NaN, n_actions::Int=0, forced::Bool=false)
    cur_mod_norm_p = 0.0
    cur_mod_norm_q = 0.0
    try
        st_now = get_state()
        cur_mods_now = _extract_current_mods(st_now)
        if !isempty(cur_mods_now)
            cur_mod_norm_p = sqrt(sum(abs2(v[1]) for v in values(cur_mods_now)))
            cur_mod_norm_q = sqrt(sum(abs2(v[2]) for v in values(cur_mods_now)))
        end
    catch
        # ignore diagnostics failure
    end
    json_value = (isfinite(cost_value) ? Float64(cost_value) : nothing)
    detail = Dict{String,Any}(
        "mode" => "julia",
        "base_max" => (isfinite(base_max) ? base_max : nothing),
        "post_max" => (isfinite(post_max) ? post_max : nothing),
        "meas_max" => (isfinite(meas_max) ? meas_max : nothing),
        "n_actions" => Int(n_actions),
        "forced" => Bool(forced),
        "cur_mod_norm_p" => cur_mod_norm_p,
        "cur_mod_norm_q" => cur_mod_norm_q,
    )
    try
        post_cost(CONTROLLER_SOURCE, json_value; detail=detail)
    catch err
        @warn "Failed posting cost" err
    end
    return nothing
end

last_iter_info = nothing
tick = 0
init_schema_from_api_with_retries!()
# store previous-iteration post-change predictions (bus => |V|)
prev_post_bus = Dict{Int,Float64}()

function render_errors(prev_post_bus::Dict{Int,Float64}, volts_mag, buses::Vector{Int})
    # Clear the terminal and print a compact table that is updated each loop.
    buf = IOBuffer()
    # ANSI: clear screen and move cursor home
    print(buf, "\u001b[2J\u001b[H")
    @printf(buf, "Time: %s  Loop period: %.3fs\n", Dates.format(now(), "HH:MM:SS"), LOOP_PERIOD_S)
    @printf(buf, "%6s %10s %10s %11s %10s\n", "Bus", "Pred(V)", "Meas(V)", "Error(P-M)", "Viol")
    for b in buses
        key = @sprintf("V%d_mag", b)
        meas = haskey(volts_mag, key) && !(volts_mag[key] === nothing) ? Float64(volts_mag[key]) : NaN
        pred = get(prev_post_bus, b, NaN)
        err = (isnan(pred) || isnan(meas)) ? NaN : (pred - meas)
        viol = isnan(meas) ? 0.0 : max(0.0, meas - COMMON_LIMIT)
        @printf(buf, "%6d %10.4f %10.4f %11.4f %10.4f\n", b, pred, meas, err, viol)
    end
    print(String(take!(buf)))
    flush(stdout)
end
while true
    t0 = time()
    enabled = try
        controller_enabled()
    catch
        false
    end
    if !enabled
        if was_enabled
            try clear_controller_mods() catch; end
        end
        global last_success_z = nothing
        global prev_post_bus = Dict{Int,Float64}()
        global was_enabled = false
        sleep(LOOP_PERIOD_S)
        continue
    else
        if !was_enabled
            try
                current_mods_seed = get_current_routed_mods()
                if !isempty(current_mods_seed)
                    seed = Float64[]
                    for bus in controlled
                        cur = get(current_mods_seed, bus, (0.0, 0.0))
                        push!(seed, cur[2])
                        push!(seed, cur[1])
                    end
                    global last_success_z = seed
                else
                    global last_success_z = nothing
                end
            catch
                global last_success_z = nothing
            end
        end
        global was_enabled = true
    end
    # 1) Pull state
    st = try
        get_state()
    catch err
        @warn "Failed to GET /state" err
        sleep(LOOP_PERIOD_S)
        continue
    end
    # 2) Update input row from service powers (MW/MVAr)
    update_input_row_from_state!(input_row_df, st)
    # 2.1) Read current mods for relative updates
    current_mods = _extract_current_mods(st)
    # 3) Measured magnitudes (from service), used for monitoring only (open loop)
    volts_mag = get(st, "voltages_mag", Dict{String,Any}())
    meas_max = try
        # restrict to predicted buses for fair comparison
        mags = Float64[]
        for b in PRED_BUS_IDS
            key = @sprintf("V%d_mag", b)
            if haskey(volts_mag, key) && !(volts_mag[key] === nothing)
                push!(mags, Float64(volts_mag[key]))
            end
        end
        isempty(mags) ? NaN : maximum(mags)
    catch
        NaN
    end
    # Build SR prediction from the API "power" snapshot (this includes the
    # actually implemented dP/dQ returned by the service). Use this as the
    # predicted voltages shown in the UI so Pred(V) aligns with measurements.
    one_nt_dbg = build_one_sample_nt(input_row_df, Vector{Symbol}(input_syms))
    println(one_nt_dbg)
    ypre = _predict_namedtuple(one_nt_dbg)
    pre_bus = Dict{Int,Float64}()
    for sym in SR_TARGETS
        m = match(r"^V(\d+)_complex$", String(sym)); bus = parse(Int, m.captures[1])
        pre_bus[bus] = get_pred_mag(ypre[sym])
    end
    # Render current SR prediction (based on API power) vs measured magnitudes
    try
        render_errors(pre_bus, volts_mag, PRED_BUS_IDS)
    catch err
        @warn "render_errors failed" err
    end
    # 4) Optimize on the PISR prediction only
    deltas_pu, base_max, post_max, post_bus, controller_cost = optimize_once!(input_row_df; current_mods=current_mods)
    # 5) Post
    maybe_post_deltas!(deltas_pu, current_mods)
    maybe_post_cost!(controller_cost; base_max=base_max, post_max=post_max, meas_max=meas_max, n_actions=length(deltas_pu))
    # pre_bus was already computed from the API 'power' snapshot above and used for rendering
    # Debug: print SR inputs and state powers for controlled buses
    if DEBUG_SR_INPUTS
        pw_cur = nothing
        try
            pw_cur = get(st, "power", nothing)
        catch
            pw_cur = nothing
        end
        if DEBUG_SR_INPUTS
            io_buf = IOBuffer()
            print(io_buf, "  SR inputs:")
            for bus in controlled
                symS = Symbol(@sprintf("S%d_complex", bus))
                val_df = symS in names(input_row_df) ? ComplexF64(input_row_df[1, symS]) : ComplexF64(0,0)
                print(io_buf, ' ', @sprintf("S%d=%s", bus, _fmt_c(val_df)))
            end
            println(String(take!(io_buf)))
            # Consolidated SR outputs pre→post for controlled buses
            io3 = IOBuffer()
            print(io3, "  SR out pre→post:")
            for bus in controlled
                pre = get(pre_bus, bus, NaN)
                post = get(post_bus, bus, NaN)
                print(io3, ' ', @sprintf("V%d=%.4f→%.4f", bus, pre, post))
            end
            println(String(take!(io3)))
            # Optional: show raw state power values for controlled buses
            if pw_cur !== nothing
                io2 = IOBuffer()
                print(io2, "  state power:")
                for bus in controlled
                    key = @sprintf("S%d_complex", bus)
                    z = _get_pw_val(pw_cur, key)
                    print(io2, ' ', @sprintf("%s=%s", key, _fmt_c(z)))
                end
                println(String(take!(io2)))
            end
        end
    end
    # Log concise summary
    max_dp = isempty(deltas_pu) ? 0.0 : maximum(abs.(first.(values(deltas_pu))))
    max_dq = isempty(deltas_pu) ? 0.0 : maximum(abs.(last.(values(deltas_pu))))
    #@printf("[rtctl] dt=%.1fms base|maxV|=%.4f post|maxV|=%.4f meas|maxV|=%s actions=%d max|ΔP|=%.5g max|ΔQ|=%.5g\n",
    #        (time()-t0)*1e3, base_max, post_max, isnan(meas_max) ? "NaN" : @sprintf("%.4f", meas_max), length(deltas_pu), max_dp, max_dq)
    # Occasional raw powers sample (every ~1s)
    global tick
    tick += 1
    if tick % Int(ceil(1/LOOP_PERIOD_S)) == 1
        # print two raw entries from powers for sanity
        try
            cnt = 0
            for (k,v) in pairs(get(st, "power", Dict{String,Any}()))
                if cnt < 2
                    #@printf("  raw %s = %s\n", String(k), String(v))
                    cnt += 1
                else
                    break
                end
            end
        catch
            # ignore
        end
    end
    # Per-bus line (suppress if DEBUG_SR_INPUTS to keep only one SR in/out block)
    if !DEBUG_SR_INPUTS
        for bus in controlled
            dp_mw, dq_mvar = get(deltas_pu, bus, (0.0, 0.0))
            # Current S input (MW/MVAr) used by SR
            sym = Symbol(@sprintf("S%d_complex", bus))
            # Read from the input_row_df we fed into the model (pre-change snapshot)
            curS = sym in names(input_row_df) ? ComplexF64(input_row_df[1, sym]) : ComplexF64(0,0)
            pre = get(pre_bus, bus, NaN)
            post = get(post_bus, bus, NaN)
            meas = try
                vm = volts_mag[bus]
                vm === nothing ? NaN : Float64(vm)
            catch
                NaN
            end
            #@printf("  bus %d: S=%.5g%+.5gj | ΔP=%.5g MW, ΔQ=%.5g MVAr | pred %.4f→%.4f meas=%.4f\n",
            #    bus, real(curS), imag(curS), dp_mw, dq_mvar, pre, post, meas)
        end
    end
    # save this iteration's post predictions for next-loop comparison
    global prev_post_bus = post_bus
    # 6) pacing
    elapsed = time() - t0
    if VERBOSE
        # already printed
    end
    sleep(max(0.0, LOOP_PERIOD_S - elapsed))
end
