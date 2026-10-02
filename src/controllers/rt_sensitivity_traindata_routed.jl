#!/usr/bin/env -S julia --project=@.

# Real-time sensitivity-based controller using only the RT API.
# - Identification phase: perturb controlled buses via /mods to estimate d|V|/dP, d|V|/dQ.
# - Control loop: when voltages violate [VMIN, VMAX], solve a small black-box optimization
#   for absolute mods (dP_mw, dQ_mvar) and POST them to /mods.
# - API calls identical in spirit to the existing RT controller: GET /status, GET /state, POST /mods.

using HTTP
using JSON3
using Printf
using Dates
using Statistics
using LinearAlgebra
using BlackBoxOptim
using Logging

# ---------------- Configuration ----------------
const PF_BASE_URL   = get(ENV, "PF_BASE_URL", "http://127.0.0.1:8000")
const CONTROLLER_SOURCE = "sensitivity"
const LOOP_PERIOD_S = try parse(Float64, get(ENV, "LOOP_PERIOD_S", "0.1")) catch; 0.1 end
const SETTLE_MS     = try parse(Int, get(ENV, "SETTLE_MS", "50")) catch; 50 end
const SLACK_BUS_ID  = try parse(Int, get(ENV, "SLACK_BUS_ID", "66")) catch; 66 end
const VMIN          = try parse(Float64, get(ENV, "VMIN", "0.97")) catch; 0.97 end
const VMAX          = try parse(Float64, get(ENV, "VMAX", get(ENV, "COMMON_V_LIMIT", "1.05"))) catch; 1.05 end

# Sensitivity finite-difference steps (MW / MVAr) used in identification
const SENS_DP_EPS_MW = try parse(Float64, get(ENV, "SENS_DP_EPS_MW", "0.05")) catch; 0.05 end
const SENS_DQ_EPS_MVAR = try parse(Float64, get(ENV, "SENS_DQ_EPS_MVAR", "0.05")) catch; 0.05 end

# Control box limits (absolute mods, MW / MVAr)
const DELTA_P_LIMIT_MW   = try parse(Float64, get(ENV, "DELTA_P_LIMIT_MW", get(ENV, "PLIMIT", "0.3"))) catch; 0.3 end
const DELTA_Q_LIMIT_MVAR = try parse(Float64, get(ENV, "DELTA_Q_LIMIT_MVAR", get(ENV, "QLIMIT", "0.1"))) catch; 0.1 end

# Objective weights and penalties
const W_P = try parse(Float64, get(ENV, "P_WEIGHT", "30.0")) catch; 30.0 end
const W_Q = try parse(Float64, get(ENV, "Q_WEIGHT", "1.0")) catch; 1.0 end
const PENALTY_W = try parse(Float64, get(ENV, "PENALTY_W", "1.0e4")) catch; 1.0e4 end

# Global optimizer knobs
const GLOB_MAXTIME   = try parse(Float64, get(ENV, "GLOB_MAXTIME", "0.02")) catch; 0.02 end
const GLOB_MAXEVALS  = try parse(Int,    get(ENV, "GLOB_MAXEVALS", "200")) catch; 200 end
const GLOB_POP       = try parse(Int,    get(ENV, "GLOB_POP", "20")) catch; 20 end
const GLOB_METHOD    = Symbol(get(ENV, "GLOB_METHOD", "de_rand_1_bin_radiuslimited"))

# Optional: explicitly set controlled buses via env (comma-separated). If not set, default to [55,57,64] and intersect with /status
function _parse_list_ints(s::AbstractString)
    parts = split(strip(s), ',')
    Int[parse(Int, p) for p in parts if !isempty(strip(p))]
end
const CONTROLLED_BUSES_ENV = [55,57,64]

const VERBOSE = lowercase(get(ENV, "VERBOSE", "false")) in ("1","true","yes","y")

# ---------------- HTTP helpers ----------------
function _http_get_json(url::String)
    r = HTTP.get(url)
    r.status == 200 || error("GET $(url) -> $(r.status)\n" * String(r.body))
    return JSON3.read(String(r.body))
end

function _http_post_json(url::String, body)
    r = HTTP.post(url; headers=Dict("Content-Type"=>"application/json"), body=JSON3.write(body))
    r.status in (200,201) || error("POST $(url) -> $(r.status)\n" * String(r.body))
    return JSON3.read(String(r.body))
end

function get_status()
    return _http_get_json(string(PF_BASE_URL, "/status"))
end

function get_state()
    return _http_get_json(string(PF_BASE_URL, "/state"))
end

function post_mods(items)
    # items :: Vector{NamedTuple{(:bus_ext,:dP_mw,:dQ_mvar),Tuple{Int,Float64,Float64}}}
    return _http_post_json(string(PF_BASE_URL, "/controller/", CONTROLLER_SOURCE, "/mods"), Dict("items" => [Dict("bus_ext"=>it.bus_ext, "dP_mw"=>it.dP_mw, "dQ_mvar"=>it.dQ_mvar) for it in items]))
end

function is_inactive_controller_conflict(err)::Bool
    msg = sprint(showerror, err)
    return occursin("409", msg) && occursin("is not active", msg)
end

function delete_controller_mods()
    r = HTTP.request("DELETE", string(PF_BASE_URL, "/controller/", CONTROLLER_SOURCE, "/mods"))
    r.status in (200, 204) || error("DELETE controller mods failed: $(r.status)")
    return true
end

function controller_enabled()
    js = _http_get_json(string(PF_BASE_URL, "/controller/", CONTROLLER_SOURCE, "/enabled"))
    return Bool(get(js, "enabled", false))
end

function post_cost(value; detail=Dict{String,Any}())
    body = Dict("source" => CONTROLLER_SOURCE, "value" => value, "detail" => detail)
    return _http_post_json(string(PF_BASE_URL, "/cost"), body)
end

function extract_current_mods(st)
    raw = haskey(st, "mods") ? st["mods"] : Dict{String,Any}()
    mods = Dict{Int,Tuple{Float64,Float64}}()
    for (k, v) in pairs(raw)
        bus = try parse(Int, String(k)) catch; continue end
        dp = try Float64(v["dP_mw"]) catch; 0.0 end
        dq = try Float64(v["dQ_mvar"]) catch; 0.0 end
        mods[bus] = (dp, dq)
    end
    return mods
end

# ---------------- Parsing helpers ----------------
function parse_complex_token(s::AbstractString)
    # Accept "+0.999-0.001j" or "(0.999-0.001j)"
    str = strip(String(s))
    if startswith(str, "(") && endswith(str, ")")
        str = str[2:end-1]
    end
    m = match(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([+-](?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)j$", str)
    if m === nothing
        return 0.0 + 0.0im
    end
    return ComplexF64(parse(Float64, m.captures[1]), parse(Float64, m.captures[2]))
end

function extract_bus_from_key(k::AbstractString, prefix::Char)
    # expects like "S57_complex" or "V64_complex"
    pat = Regex("^" * string(prefix) * "(\\d+)_complex\\z")
    m = match(pat, k)
    return m === nothing ? nothing : parse(Int, m.captures[1])
end

function voltages_mag_from_state(st)
    # Prefer complex voltages -> compute magnitude; fallback to voltages_mag if present
    if haskey(st, "voltages")
        mags = Dict{Int,Float64}()
        for (k,v) in pairs(st["voltages"])
            bus = extract_bus_from_key(String(k), 'V')
            bus === nothing && continue
            z = try
                parse_complex_token(String(v))
            catch
                0.0 + 0.0im
            end
            mags[bus] = abs(z)
        end
        return mags
    elseif haskey(st, "voltages_mag")
        mags = Dict{Int,Float64}()
        for (k,v) in pairs(st["voltages_mag"])
            # keys may be like "V57" or "57"; extract first integer sequence
            m = match(r"(\d+)", String(k))
            m === nothing && continue
            mags[parse(Int, m.captures[1])] = try
                v isa Number ? float(v) : parse(Float64, String(v))
            catch; NaN; end
        end
        return mags
    else
        return Dict{Int,Float64}()
    end
end

function power_map_from_state(st)
    # Returns Dict{Int,ComplexF64} of totals used in next PF step
    pw = haskey(st, "power") ? st["power"] : (haskey(st, "power_orig") ? st["power_orig"] : nothing)
    pw === nothing && return Dict{Int,ComplexF64}()
    out = Dict{Int,ComplexF64}()
    for (k,v) in pairs(pw)
        bus = extract_bus_from_key(String(k), 'S')
        bus === nothing && continue
        out[bus] = try
            parse_complex_token(String(v))
        catch
            0.0 + 0.0im
        end
    end
    return out
end

# ---------------- Identification: load offline training-based sensitivities ----------------
# We'll load a training CSV, build Δ-based H and Gamma matrices and compute
# regularized LS sensitivities + 3σ uncertainties (same method as offline MPC script).
const TRAIN_COMPLEX_CSV = get(ENV, "TRAIN_COMPLEX_CSV", joinpath(normpath(joinpath(@__DIR__, "..", "..")), "models", "cil_longrun", "train_data_complex.csv"))

function parse_py_complex(str::AbstractString)
    s = replace(strip(String(str)), r"\s+" => "")
    startswith(s, "(") && endswith(s, ")") && (s = s[2:end-1])
    m = match(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([+-](?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)j$", s)
    m === nothing && return 0.0 + 0.0im
    return ComplexF64(parse(Float64, m.captures[1]), parse(Float64, m.captures[2]))
end

function load_complex_df(path::AbstractString)
    isfile(path) || error("CSV not found: $path")
    txt = read(path, String)
    lines = split(txt, '\n')
    isempty(lines) && error("Empty file: $path")
    header = Symbol.(split(strip(lines[1]), ','))
    body = join(lines[2:end], "\n")
    token_pattern = r"\([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[+-](?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?j\)"
    tokens = [m.match for m in eachmatch(token_pattern, body)]
    n_cols = length(header)
    length(tokens) % n_cols == 0 || error("token count not divisible by columns")
    n_rows = length(tokens) ÷ n_cols
    df = Dict{Symbol, Vector{ComplexF64}}()
    for (j, h) in enumerate(header)
        col = Vector{ComplexF64}(undef, n_rows)
        for i in 1:n_rows
            col[i] = parse_py_complex(tokens[(i-1)*n_cols + j])
        end
        df[Symbol(h)] = col
    end
    # Convert to simple NamedTuple-like object using Dict to avoid adding DataFrames dependency
    return df, n_rows, header
end

function build_H_Gamma_from_train_dict(train_dict::Dict{Symbol, Vector{ComplexF64}}, nrows::Int, slack_bus::Int)
    # collect column symbols
    keys_sym = collect(keys(train_dict))
    voltage_cols = [k for k in keys_sym if occursin(r"^V\d+_complex$", String(k))]
    power_cols   = [k for k in keys_sym if occursin(r"^S\d+_complex$", String(k))]
    isempty(voltage_cols) && error("No voltage columns found in training data")
    isempty(power_cols) && error("No power columns found in training data")

    slack_sym = Symbol("V$(slack_bus)_complex")
    target_voltage_cols = [c for c in voltage_cols if c != slack_sym]
    n_targets = length(target_voltage_cols)
    controlled_S_syms = power_cols
    n_ctrl = length(controlled_S_syms)

    N = nrows
    @assert N >= 2 "Training data needs at least two time samples"
    n_diffs = N - 1
    H = zeros(Float64, n_diffs, 2 * n_ctrl)
    Gamma = zeros(Float64, n_diffs, n_targets)

    for t in 2:N
        ri = t - 1
        for (j, s_sym) in enumerate(controlled_S_syms)
            s_now = train_dict[s_sym][t]; s_prev = train_dict[s_sym][t-1]
            ΔP = real(s_now) - real(s_prev)
            ΔQ = imag(s_now) - imag(s_prev)
            H[ri, j] = ΔP
            H[ri, n_ctrl + j] = ΔQ
        end
        for (i, v_sym) in enumerate(target_voltage_cols)
            v_now = train_dict[v_sym][t]; v_prev = train_dict[v_sym][t-1]
            Gamma[ri, i] = abs(v_now) - abs(v_prev)
        end
    end
    return H, Gamma, controlled_S_syms, target_voltage_cols
end

function estimate_sensitivities_and_uncertainties(H::Matrix{Float64}, Gamma::Matrix{Float64}; lambda::Float64=1e-6)
    n_rows, n_features = size(H)
    n_targets = size(Gamma, 2)
    R = transpose(H) * H + lambda * I(n_features)
    X_hat = R \ (transpose(H) * Gamma)
    Resid = Gamma - H * X_hat
    dof = max(1, n_rows - n_features)
    sigma_r = sqrt.(sum(Resid .^ 2, dims=1)[:] ./ dof)
    Rinv = inv(R)
    diagRinv = diag(Rinv)
    coeff_std_base = sqrt.(diagRinv)
    coeff_sigmas = zeros(Float64, n_features, n_targets)
    for j in 1:n_targets
        coeff_sigmas[:, j] = sigma_r[j] .* coeff_std_base
    end
    n_ctrl = n_features ÷ 2
    Kp = transpose( X_hat[1:n_ctrl, :] )
    Kq = transpose( X_hat[n_ctrl+1:end, :] )
    sig_p = transpose( coeff_sigmas[1:n_ctrl, :] )
    sig_q = transpose( coeff_sigmas[n_ctrl+1:end, :] )
    ΔKp = 3.0 .* sig_p
    ΔKq = 3.0 .* sig_q
    return Kp, Kq, ΔKp, ΔKq
end

function get_controlled_buses_from_training(header)
    # header is Vector{String} of column names; find S*_complex columns and return their bus ints
    scols = [String(h) for h in header if occursin(r"^S\d+_complex$", String(h))]
    buses = Int[]
    for s in scols
        m = match(r"S(\d+)_complex", s)
        m !== nothing && push!(buses, parse(Int, m.captures[1]))
    end
    return sort(unique(buses))
end

# ---------------- Control loop ----------------
function run_control_loop()
    # Controlled set: try env or infer from training file
    println("[sens-rt] Loading offline sensitivities from training CSV: ", TRAIN_COMPLEX_CSV)
    train_dict, nrows, header = load_complex_df(TRAIN_COMPLEX_CSV)
    controlled = isempty(CONTROLLED_BUSES_ENV) ? get_controlled_buses_from_training(header) : CONTROLLED_BUSES_ENV
    println(@sprintf("[sens-rt] Controlled buses: %s", join(string.(controlled), ", ")))
    # Build H/Gamma and estimate sensitivities
    H, Gamma, controlled_S_syms, target_voltage_cols = build_H_Gamma_from_train_dict(train_dict, nrows, SLACK_BUS_ID)
    Kp, Kq, ΔKp, ΔKq = estimate_sensitivities_and_uncertainties(H, Gamma; lambda=1e-6)
    println(@sprintf("[sens-rt] Loaded Kp/Kq sizes: %dx%d", size(Kp,1), size(Kp,2)))
    if size(Kp,1) > 0 && size(Kp,2) > 0
        println(@sprintf("[sens-rt] Example Kp[1,1]=%.6e ΔKp[1,1]=%.6e", Kp[1,1], ΔKp[1,1]))
    end

    # Maintain last applied absolute mods (so we can reapply each loop). Start at zeros.
    mods_state_P = Dict(b => 0.0 for b in controlled)
    mods_state_Q = Dict(b => 0.0 for b in controlled)

    # Utility to build |V| vector in target order
    function read_Vmag_targets()
        st = get_state()
        vm = voltages_mag_from_state(st)
        return [get(vm, b, NaN) for b in target_buses], st
    end

    # Indices map for controlled buses (map bus number to column index in Kp/Kq)
    # controlled_S_syms are like Symbol("S55_complex") etc.; build mapping from bus->index
    ctrl_map = Dict{Int,Int}()
    for (j, s_sym) in enumerate(controlled_S_syms)
        m = match(r"S(\d+)_complex", String(s_sym))
        m !== nothing && (ctrl_map[parse(Int, m.captures[1])] = j)
    end
    ctrl_idx = Dict{Int,Int}()
    for (i, b) in enumerate(controlled)
        ctrl_idx[b] = haskey(ctrl_map, b) ? ctrl_map[b] : i
    end

    # Optimization variables are absolute mods (in MW/MVAr) to set at the service
    target_buses = [parse(Int, replace(String(v), r"V|_complex" => "")) for v in target_voltage_cols]
    n_targets = length(target_buses)
    n = length(controlled)
    was_enabled = false

    # Main loop
    while true
        loop_t0 = time()

        enabled = try
            controller_enabled()
        catch
            false
        end
        if !enabled
            if was_enabled
                try delete_controller_mods() catch; end
            end
            for b in controlled
                mods_state_P[b] = 0.0
                mods_state_Q[b] = 0.0
            end
            was_enabled = false
            sleep(max(0.0, LOOP_PERIOD_S - (time() - loop_t0)))
            continue
        elseif !was_enabled
            try
                st_seed = get_state()
                seed_mods = extract_current_mods(st_seed)
                for b in controlled
                    cur = get(seed_mods, b, (0.0, 0.0))
                    mods_state_P[b] = cur[1]
                    mods_state_Q[b] = cur[2]
                end
            catch
                for b in controlled
                    mods_state_P[b] = 0.0
                    mods_state_Q[b] = 0.0
                end
            end
            was_enabled = true
        end

        Vcurr_vec, st = read_Vmag_targets()
        base_max = maximum(Vcurr_vec)
        live_mods = extract_current_mods(st)
        for b in controlled
            cur = get(live_mods, b, (0.0, 0.0))
            mods_state_P[b] = cur[1]
            mods_state_Q[b] = cur[2]
        end

        # NOTE: Match PISR convention: dP_mw is always non-negative (absolute magnitude).

        over_volt = base_max > VMAX
        under_volt = minimum(Vcurr_vec) < VMIN
        need = over_volt || under_volt

        if VERBOSE
            println(@sprintf("[sens-rt] t=%s base_max=%.4f need=%s", Dates.format(now(), "HH:MM:SS"), base_max, string(need)))
        end

    # Always optimize: if no violation, objective will push mods toward 0 while keeping voltages within bounds.

        # Objective: cost + penalties using sensitivity model on RELATIVE changes from current mods
        function objective_vec(z)
            dP_abs = @view z[1:n]
            dQ_abs = @view z[n+1:2*n]
            # relative change vs current applied mods for the linear prediction
            dP = similar(dP_abs)
            dQ = similar(dQ_abs)
            @inbounds for k in 1:n
                b = controlled[k]
                dP[k] = dP_abs[k] - mods_state_P[b]
                dQ[k] = dQ_abs[k] - mods_state_Q[b]
            end
            # penalize absolute mods to encourage returning to zero when feasible
            cost = W_P * sum(abs2, dP_abs) + W_Q * sum(abs2, dQ_abs)
            over_pen = 0.0
            under_pen = 0.0
            # Use estimated Kp/Kq and ΔKp/ΔKq; mapping indices: K matrices are ordered per controlled_S_syms
            for i in 1:n_targets
                # build per-controller d vectors in the order of controlled_S_syms
                dP_ordered = zeros(Float64, size(Kp,2))
                dQ_ordered = zeros(Float64, size(Kq,2))
                for (k, b) in enumerate(controlled)
                    idx = ctrl_idx[b]
                    if idx <= length(dP_ordered)
                        dP_ordered[idx] = dP[k]
                        dQ_ordered[idx] = dQ[k]
                    end
                end
                dv_lin = dot(Kp[i, :], dP_ordered) + dot(Kq[i, :], dQ_ordered)
                robust_margin = sum(abs.(dP_ordered) .* ΔKp[i, :]) + sum(abs.(dQ_ordered) .* ΔKq[i, :])
                vhat_up = Vcurr_vec[i] + dv_lin + robust_margin
                vhat_lo = Vcurr_vec[i] + dv_lin - robust_margin
                if vhat_up > VMAX
                    over_pen += (vhat_up - VMAX)^2
                end
                if vhat_lo < VMIN
                    under_pen += (VMIN - vhat_lo)^2
                end
            end
            return cost + PENALTY_W * (over_pen + under_pen)
        end

        # Variable ranges (absolute mods to set).
        ranges = Vector{Tuple{Float64,Float64}}(undef, 2*n)
        for j in 1:n
            ranges[j] = (0.0, DELTA_P_LIMIT_MW)
        end
        for j in 1:n
            ranges[n+j] = (-DELTA_Q_LIMIT_MVAR, DELTA_Q_LIMIT_MVAR)
        end

        # Build an initial vector inside ranges (use current mods as seed)
        init_z = Vector{Float64}(undef, 2*n)
        for j in 1:n
            base = mods_state_P[controlled[j]]
            lo, hi = ranges[j]
            init_z[j] = clamp(base, lo, hi)
        end
        for j in 1:n
            baseq = mods_state_Q[controlled[j]]
            lo, hi = ranges[n+j]
            init_z[n+j] = clamp(baseq, lo, hi)
        end

        res = bboptimize(objective_vec, init_z;
            SearchRange = ranges,
            NumDimensions = 2*n,
            Method = GLOB_METHOD,
            PopulationSize = GLOB_POP,
            MaxFuncEvals = GLOB_MAXEVALS,
            MaxTime = GLOB_MAXTIME,
            TraceMode = :silent,
            Seed = 42,
        )
        best_z = best_candidate(res)
        zopt = best_z === nothing ? init_z : best_z
        objective_best = try
            Float64(objective_vec(zopt))
        catch
            NaN
        end
    dP_sol = clamp.(zopt[1:n], map(x->x[1], ranges[1:n]), map(x->x[2], ranges[1:n]))
    dQ_sol = clamp.(zopt[n+1:2*n], map(x->x[1], ranges[n+1:2*n]), map(x->x[2], ranges[n+1:2*n]))

        # Sanitize
        if any(x->!isfinite(x), dP_sol) || any(x->!isfinite(x), dQ_sol)
            dP_sol = map(x->isfinite(x) ? x : 0.0, dP_sol)
            dQ_sol = map(x->isfinite(x) ? x : 0.0, dQ_sol)
        end

        # Post absolute mods
        items = Vector{NamedTuple{(:bus_ext,:dP_mw,:dQ_mvar),Tuple{Int,Float64,Float64}}}(undef, n)
        for (k, bus) in enumerate(controlled)
            items[k] = (bus_ext=bus, dP_mw=dP_sol[k], dQ_mvar=dQ_sol[k])
        end
        # log a tiny summary including robust margins for first few targets
        if VERBOSE
            println(@sprintf("[sens-rt] Posting mods (first 4 P): %s", join(string.(round.(dP_sol[1:min(4,end)], digits=4)), ", ")))
        end
        try
            post_mods(items)
            # Update our local mod state
            for (k, bus) in enumerate(controlled)
                mods_state_P[bus] = dP_sol[k]
                mods_state_Q[bus] = dQ_sol[k]
            end
            try
                post_cost(isfinite(objective_best) ? objective_best : nothing; detail=Dict(
                    "mode" => CONTROLLER_SOURCE,
                    "base_max" => base_max,
                    "n_actions" => n,
                    "need" => need,
                ))
            catch
                # ignore cost post failures
            end
        catch e
            if is_inactive_controller_conflict(e)
                sleep(max(0.0, LOOP_PERIOD_S - (time() - loop_t0)))
                continue
            end
            @warn "POST /mods failed" e
        end

        sleep(max(0.0, LOOP_PERIOD_S - (time() - loop_t0)))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println(@sprintf("[sens-rt] Starting controller. Loop=%.0f ms", LOOP_PERIOD_S*1e3))
    try
        run_control_loop()
    catch e
        # Print a compact error and exit non-zero
        println("[sens-rt] ERROR: ", e)
        Base.exit(1)
    end
end
