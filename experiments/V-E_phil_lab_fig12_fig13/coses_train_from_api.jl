#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
Collect data from the realtime powerflow API, build train/test datasets, and train/evaluate a SR model.

What this does now (system ID focused):
 1) Collect 50 training samples while injecting a configurable random dQ signal via /mods.
 2) Collect 20 test samples using a block-rand scheme: each load gets a block with random dP/dQ.
 3) Save both CSVs with Python-style complex tokens `(a+bj)`.
 4) Train on the 50-sample set and verify performance on the 20-sample test set using incremental SR.

Key environment variables (or pass as CLI KEY=VALUE):
    SERVICE_URL           default: http://127.0.0.1:8000
    POLL_MS               default: 200         # ms between samples
    SETTLE_MS             default: 150         # ms to wait after posting /mods before sampling
    SLACK_BUS_ID          default: 66          # must be present in collected V* columns
    TRAIN_SAMPLES         default: 50
    TRAIN_DQ_MAX_MVAR     default: 0.03        # amplitude for reactive power perturbations
    TRAIN_DQ_SAMPLES      default: 20          # how many of the 50 samples receive dQ perturbations
    TRAIN_DQ_MODE         default: all         # all | one  (apply dQ to all controlled loads or one random)
    TRAIN_SEED            default: 0
    TEST_SAMPLES          default: 20
    TEST_DP_MAX_MW        default: 0.03        # amplitude for active power blockrand
    TEST_DQ_MAX_MVAR      default: 0.03        # amplitude for reactive power blockrand
    TEST_SEED             default: 0
    INITIAL_ITER          default: 10
    STEP_ITER             default: 5
    MAX_TOTAL_ITER        default: 120
    THRESHOLD             default: 0.01
    MAX_SIZE              default: 30
    NP_FACTOR             default: 3
    MULTIPROCESS          default: false
    OUTPUT_DIR            default: auto under results/V-E_phil_lab_fig12_fig13/

Usage:
    julia --project=. --threads auto experiments/V-E_phil_lab_fig12_fig13/coses_train_from_api.jl TRAIN_SAMPLES=50 TEST_SAMPLES=20
"""
# julia --project=. --threads auto experiments/V-E_phil_lab_fig12_fig13/coses_train_from_api.jl   (lab PC: Windows, Julia 1.11.6)
using Dates
using Printf
using DataFrames
using Statistics
using HTTP
using JSON3
using CSV
using Random

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

# Define a wrapper module to include training script without running its main()
module IncWrap
    isinteractive() = true
end

"""
Parse KEY=VALUE CLI args into a Dict for easy overrides.
"""
function parse_kv_args!(dict::Dict{String,String})
    for a in ARGS
        if occursin('=', a)
            k, v = split(a, "=", limit=2)
            dict[k] = v
        end
    end
    return dict
end

function getenv_or(args::Dict{String,String}, key::String, default)
    haskey(args, key) && return args[key]
    return get(ENV, key, default)
end

_raw_args = parse_kv_args!(Dict{String,String}())

SERVICE_URL         = getenv_or(_raw_args, "SERVICE_URL", "http://127.0.0.1:8000")
POLL_MS             = parse(Int, getenv_or(_raw_args, "POLL_MS", "1000"))
SETTLE_MS           = parse(Int, getenv_or(_raw_args, "SETTLE_MS", "700"))
SLACK_BUS_ID        = parse(Int, getenv_or(_raw_args, "SLACK_BUS_ID", "66"))
TRAIN_SAMPLES       = parse(Int, getenv_or(_raw_args, "TRAIN_SAMPLES", "100"))
TRAIN_DQ_MAX_MVAR   = parse(Float64, getenv_or(_raw_args, "TRAIN_DQ_MAX_MVAR", "2000"))
TRAIN_DP_MAX_MW   = parse(Float64, getenv_or(_raw_args, "TRAIN_DP_MAX_MW", "2000"))
TRAIN_DQ_SAMPLES    = parse(Int, getenv_or(_raw_args, "TRAIN_DQ_SAMPLES", "100"))
TRAIN_DQ_MODE       = lowercase(string(getenv_or(_raw_args, "TRAIN_DQ_MODE", "all")))  # all|one
TRAIN_SEED          = parse(Int, getenv_or(_raw_args, "TRAIN_SEED", "0"))

TEST_SAMPLES        = parse(Int, getenv_or(_raw_args, "TEST_SAMPLES", "20"))
TEST_DP_MAX_MW      = parse(Float64, getenv_or(_raw_args, "TEST_DP_MAX_MW", "2000"))
TEST_DQ_MAX_MVAR    = parse(Float64, getenv_or(_raw_args, "TEST_DQ_MAX_MVAR", "2000"))
TEST_SEED           = parse(Int, getenv_or(_raw_args, "TEST_SEED", "0"))
INITIAL_ITER        = parse(Int, getenv_or(_raw_args, "INITIAL_ITER", "20"))
STEP_ITER           = parse(Int, getenv_or(_raw_args, "STEP_ITER", "100"))
MAX_TOTAL_ITER      = parse(Int, getenv_or(_raw_args, "MAX_TOTAL_ITER", "1000"))
THRESHOLD           = parse(Float64, getenv_or(_raw_args, "THRESHOLD", "0.001"))
MAX_SIZE            = parse(Int, getenv_or(_raw_args, "MAX_SIZE", "30"))
NP_FACTOR           = parse(Int, getenv_or(_raw_args, "NP_FACTOR", "3"))
MULTIPROCESS        = lowercase(string(getenv_or(_raw_args, "MULTIPROCESS", "false"))) in ("1","true","yes")
OUTPUT_DIR          = getenv_or(_raw_args, "OUTPUT_DIR", "")

@info "Config (collect_and_train_from_api)" SERVICE_URL POLL_MS SETTLE_MS SLACK_BUS_ID TRAIN_SAMPLES TRAIN_DP_MAX_MW TRAIN_DQ_MAX_MVAR TRAIN_DQ_SAMPLES TRAIN_DQ_MODE TRAIN_SEED TEST_SAMPLES TEST_DP_MAX_MW TEST_DQ_MAX_MVAR TEST_SEED INITIAL_ITER STEP_ITER MAX_TOTAL_ITER THRESHOLD MAX_SIZE NP_FACTOR MULTIPROCESS OUTPUT_DIR

"""
Parse a complex string that may look like "+0.999-0.001j" or "(0.999-0.001j)" to ComplexF64.
Falls back to 0+0j on malformed input.
"""
function parse_complex_str(s::AbstractString)
    str = strip(String(s))
    if isempty(str)
        return 0.0 + 0.0im
    end
    if startswith(str, "(") && endswith(str, ")")
        str = str[2:end-1]
    end
    m = match(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([+-](?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)j$", str)
    if m === nothing
        return 0.0 + 0.0im
    end
    re_s, im_s = m.captures
    return ComplexF64(parse(Float64, re_s), parse(Float64, im_s))
end

"""
Format ComplexF64 to Python-style token "(a+bj)" with high precision.
"""
_cx_to_pystr(z::ComplexF64) = @sprintf("(%0.16f%+0.16fj)", real(z), imag(z))

"""
Fetch JSON from url and return parsed object.
"""
function get_json(url::String)
    r = HTTP.get(url)
    if r.status != 200
        error("HTTP GET $(url) -> status $(r.status)")
    end
    return JSON3.read(String(r.body))
end

"""
POST /mods with a list of items [{bus_ext, dP_mw, dQ_mvar}...].
"""
function post_mods(service_url::String, items::Vector{<:NamedTuple})
    body = JSON3.write(Dict("items" => [Dict("bus_ext"=>it.bus_ext, "dP_mw"=>it.dP_mw, "dQ_mvar"=>it.dQ_mvar) for it in items]))
    r = HTTP.post(string(service_url, "/mods"); headers=Dict("Content-Type"=>"application/json"), body=body)
    r.status in (200,201) || @warn "POST /mods failed" status=r.status body=String(r.body)
    return r.status in (200,201)
end

"""
DELETE /mods to clear all mods.
"""
function clear_mods(service_url::String)
    r = HTTP.request("DELETE", string(service_url, "/mods"))
    r.status == 200 || @warn "DELETE /mods failed" status=r.status body=String(r.body)
    return r.status == 200
end

"""
GET /status and return Dict with controlled buses.
"""
function get_controlled_buses(service_url::String)
    js = get_json(string(service_url, "/status"))
    if haskey(js, "controlled_buses")
        return Vector{Int}(js["controlled_buses"])
    else
        @warn "/status missing controlled_buses; falling back to /buses"
        jb = get_json(string(service_url, "/buses"))
        return Vector{Int}(jb["controlled"])
    end
end

"""
Collect N samples from /state, sleeping POLL_MS between polls.
Returns a DataFrame with ComplexF64 columns for S*_complex and V*_complex.
Note on timing alignment:
    - /state voltages reflect the last completed PF step ("last_voltages" in service).
    - /state exposes both powers_last (PF inputs used for that last step) and powers (totals for the next step).
    - We therefore PREFER powers_last when present to align S and V from the same PF step;
        otherwise we fall back to powers.
"""
function collect_samples(service_url::String; n_samples::Int, poll_ms::Int, settle_ms::Int=150, schedule_mods::Function = _->Vector{NamedTuple}(), controlled_buses::Vector{Int}=Int[])
    rows = Vector{Dict{Symbol,ComplexF64}}(undef, n_samples)
    all_V_syms = Set{Symbol}()
    all_S_syms = Set{Symbol}()
    for i in 1:n_samples
        # Apply per-sample modifications (always set all controlled buses explicitly)
        items = schedule_mods(i)
        if isempty(items) && !isempty(controlled_buses)
            items = [ (bus_ext=b, dP_mw=0.0, dQ_mvar=0.0) for b in controlled_buses ]
        end
        if !isempty(items)
            post_mods(service_url, items)
            sleep(max(0, settle_ms)/1000)
        end
        # Fetch /state ensuring we get powers_last to align S and V from the SAME PF step
        js = nothing
        begin
            local tries = 0
            while tries < 10
                tries += 1
                local tmp = get_json(string(service_url, "/state"))
                if haskey(tmp, "power")
                    js = tmp
                    break
                end
                # If service hasn't produced a new PF step yet, wait a bit and retry
                sleep(poll_ms/1000)
            end
            js === nothing && error("/state did not provide powers after retries; increase SETTLE_MS/POLL_MS or check service rate")
        end
        vmap = Dict{Symbol,ComplexF64}()
        if haskey(js, "voltages")
            for (k,v) in pairs(js["voltages"])
                sym = Symbol(String(k))
                z = parse_complex_str(String(v))
                vmap[sym] = z
                push!(all_V_syms, sym)
            end
        end
        # Use PF inputs used for the last completed step ONLY for alignment
        for (k,v) in pairs(js["power"])
            sym = Symbol(String(k))
            z = parse_complex_str(String(v))
            vmap[sym] = z
            push!(all_S_syms, sym)
        end
        if haskey(js, "pv_power") && haskey(js["pv_power"], "SPV_complex")
            sym = :SPV_complex
            z = parse_complex_str(String(js["pv_power"]["SPV_complex"]))
            vmap[sym] = z
            push!(all_S_syms, sym)
        end
        rows[i] = vmap
        sleep(poll_ms/1000)
    end
    # Stable ordered columns: S by bus id, then V by bus id
    function sort_syms_by_bus(syms::Vector{Symbol}, prefix::Char)
        pat = Regex("^" * string(prefix) * "(\\d+)_complex" * "\$")
        sort(syms; by = s -> try
            m = match(pat, String(s))
            m === nothing ? typemax(Int) : parse(Int, m.captures[1])
        catch; typemax(Int); end)
    end
    S_cols = sort_syms_by_bus(collect(all_S_syms), 'S')
    V_cols = sort_syms_by_bus(collect(all_V_syms), 'V')
    # Build DataFrame
    df = DataFrame()
    for c in S_cols
        df[!, c] = [get(rows[i], c, 0.0+0.0im) for i in 1:n_samples]
    end
    for c in V_cols
        df[!, c] = [get(rows[i], c, 0.0+0.0im) for i in 1:n_samples]
    end
    return df
end

"""
Save complex DataFrame to CSV with Python-style complex tokens per cell.
"""
function save_complex_csv(path::AbstractString, df::DataFrame)
    str_df = DataFrame()
    for c in names(df)
        str_df[!, c] = [_cx_to_pystr(ComplexF64(v)) for v in df[!, c]]
    end
    CSV.write(path, str_df)
    return path
end

"""
Choose a slack bus id: prefer ENV/CLI SLACK_BUS_ID if present in V columns; otherwise
fallback to the numerically smallest V bus id.
"""
function choose_slack_bus(df::DataFrame, preferred::Int)
    vcols = [String(c) for c in names(df) if occursin(r"^V\d+_complex$", String(c))]
    buses = sort([parse(Int, match(r"^V(\d+)_complex$", c).captures[1]) for c in vcols])
    isempty(buses) && error("No V*_complex columns found in collected data")
    if preferred in buses
        return preferred
    else
        @warn "Preferred SLACK_BUS_ID not in dataset; using first available" preferred available_first=buses[1]
        return buses[1]
    end
end

"""
Create a per-sample schedule function for training dP/dQ injection.
Returns f(i)::Vector{NamedTuple{(:bus_ext,:dP_mw,:dQ_mvar),Tuple{Int,Float64,Float64}}}.
"""
function make_training_perturb_schedule(controlled::Vector{Int}; n_samples::Int, dp_max::Float64, dq_max::Float64, n_perturb::Int, seed::Int, mode::String)
    Random.seed!(seed)
    n_perturb = clamp(n_perturb, 0, n_samples)
    pert_idx = sort!(collect(Iterators.take(Random.shuffle(1:n_samples), n_perturb)))
    pert_set = Set(pert_idx)
    if mode == "one"
        picks = [controlled[rand(1:length(controlled))] for _ in 1:n_perturb]
    else
        picks = fill(-1, n_perturb)  # -1 => all
    end
    pick_map = Dict(pert_idx[k] => picks[k] for k in 1:n_perturb)
    return function(i::Int)
        if !(i in pert_set)
            return NamedTuple{(:bus_ext,:dP_mw,:dQ_mvar)}[(bus_ext=b, dP_mw=0.0, dQ_mvar=0.0) for b in controlled]
        end
        tgt = pick_map[i]
        if tgt == -1
            return NamedTuple{(:bus_ext,:dP_mw,:dQ_mvar)}[(bus_ext=b, dP_mw=(2rand()-1)*dp_max, dQ_mvar=(2rand()-1)*dq_max) for b in controlled]
        else
            items = NamedTuple{(:bus_ext,:dP_mw,:dQ_mvar)}[(bus_ext=b, dP_mw=0.0, dQ_mvar=0.0) for b in controlled]
            # set only one
            for idx in 1:length(items)
                if items[idx].bus_ext == tgt
                    items[idx] = (bus_ext=tgt, dP_mw=(2rand()-1)*dp_max, dQ_mvar=(2rand()-1)*dq_max)
                    break
                end
            end
            return items
        end
    end
end

"""
Create block-rand schedule for test set, following notebook logic: first block idle, then each load gets a block.
"""
function make_blockrand_schedule(controlled::Vector{Int}; n_samples::Int, dp_max::Float64, dq_max::Float64, seed::Int)
    Random.seed!(seed)
    n_loads = length(controlled)
    block_size = max(1, fld(n_samples, n_loads + 1))
    # Map sample index -> bus that gets randomized in that block (or 0 for idle)
    function which_bus_for_sample(i::Int)
        blk = fld(i-1, block_size)  # 0-based block index
        if blk == 0
            return 0
        end
        bus_idx = blk
        if bus_idx >= 1 && bus_idx <= n_loads
            return controlled[bus_idx]
        else
            return 0
        end
    end
    return function(i::Int)
        tgt = which_bus_for_sample(i)
        if tgt == 0
            return NamedTuple{(:bus_ext,:dP_mw,:dQ_mvar)}[(bus_ext=b, dP_mw=0.0, dQ_mvar=0.0) for b in controlled]
        else
            return NamedTuple{(:bus_ext,:dP_mw,:dQ_mvar)}[
                (bus_ext=b, dP_mw=(b==tgt ? (2rand()-1)*dp_max : 0.0), dQ_mvar=(b==tgt ? (2rand()-1)*dq_max : 0.0))
                for b in controlled
            ]
        end
    end
end

function main()
    println("Connecting to service at ", SERVICE_URL, " ...")
    controlled = get_controlled_buses(SERVICE_URL)
    @info "Controlled buses" controlled
    #controlled = filter(b -> b != 6, controlled)
    #@info "Filtered controlled buses" controlled

    # 1) TRAIN collection with random dP/dQ injection
    train_sched = make_training_perturb_schedule(controlled; n_samples=TRAIN_SAMPLES, dp_max=TRAIN_DP_MAX_MW, dq_max=TRAIN_DQ_MAX_MVAR, n_perturb=TRAIN_DQ_SAMPLES, seed=TRAIN_SEED, mode=TRAIN_DQ_MODE)
    println(@sprintf("Collecting %d training samples with dP/dQ injection (max dP=%.3f MW, max dQ=%.3f MVAr, n_pert=%d, mode=%s)...", TRAIN_SAMPLES, TRAIN_DP_MAX_MW, TRAIN_DQ_MAX_MVAR, TRAIN_DQ_SAMPLES, TRAIN_DQ_MODE))
    clear_mods(SERVICE_URL)
    train_df = collect_samples(SERVICE_URL; n_samples=TRAIN_SAMPLES, poll_ms=POLL_MS, settle_ms=SETTLE_MS, schedule_mods=train_sched, controlled_buses=controlled)
    clear_mods(SERVICE_URL)
    @info @sprintf("Collected TRAIN: %d samples, %d columns", nrow(train_df), ncol(train_df))

    # 2) TEST collection with block-rand P/Q
    test_sched = make_blockrand_schedule(controlled; n_samples=TEST_SAMPLES, dp_max=TEST_DP_MAX_MW, dq_max=TEST_DQ_MAX_MVAR, seed=TEST_SEED)
    println(@sprintf("Collecting %d test samples with block-rand (dP<=%.3f MW, dQ<=%.3f MVAr)...", TEST_SAMPLES, TEST_DP_MAX_MW, TEST_DQ_MAX_MVAR))
    clear_mods(SERVICE_URL)
    test_df = collect_samples(SERVICE_URL; n_samples=TEST_SAMPLES, poll_ms=POLL_MS, settle_ms=SETTLE_MS, schedule_mods=test_sched, controlled_buses=controlled)
    clear_mods(SERVICE_URL)
    @info @sprintf("Collected TEST: %d samples, %d columns", nrow(test_df), ncol(test_df))

    # Output directory and CSVs
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out_base = isempty(OUTPUT_DIR) ? joinpath(REPO_ROOT, "results", "V-E_phil_lab_fig12_fig13", @sprintf("collect_train_%s", timestamp)) : OUTPUT_DIR
    mkpath(out_base)
    train_csv = joinpath(out_base, "train_data_complex.csv")
    test_csv  = joinpath(out_base, "test_data_complex_blockrand.csv")
    save_complex_csv(train_csv, train_df)
    save_complex_csv(test_csv,  test_df)
    println("Saved TRAIN to ", train_csv)
    println("Saved TEST  to ", test_csv)

    # Bring in the incremental training routine without triggering its main()
    inc_path = joinpath(REPO_ROOT, "src", "pisr", "incremental_noslack.jl")
    isfile(inc_path) || error("Missing incremental training script: " * inc_path)
    Base.include(IncWrap, inc_path)

    # Decide slack bus from collected data (use training set)
    slack_bus = choose_slack_bus(train_df, SLACK_BUS_ID)
    println(@sprintf("Using slack bus: %d", slack_bus))

    # Train on TRAIN, evaluate on TEST
    out_dir = joinpath(out_base, "training")
    result = Base.invokelatest(() -> IncWrap.incremental_train(train_df, test_df; slack_bus=slack_bus, initial_iter=INITIAL_ITER, step_iter=STEP_ITER, max_total_iter=MAX_TOTAL_ITER, threshold=THRESHOLD, max_size=MAX_SIZE, np_factor=NP_FACTOR, multiprocess=MULTIPROCESS, output_dir=out_dir, save_snapshots=false))
    mach, metrics_vec, _ = result

    final_iter = metrics_vec[end].iteration
    final_model_path = joinpath(out_dir, @sprintf("final_model_iter%04d.jls", final_iter))
    println("Training complete. Final model: ", final_model_path)
    println(@sprintf("Final test set errors — Mean |V| MAE: %.6f, Worst |V| Max: %.6f", metrics_vec[end].mean_mag_mae, metrics_vec[end].worst_mag_max))
    println("To use with rt_controller, set MODEL_PATH=", final_model_path)
end

isinteractive() || main()
