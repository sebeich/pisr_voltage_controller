#!/usr/bin/env julia
# -*- coding: utf-8 -*-

"""
Evaluate one serialized SR model against multiple test CSV datasets in a single Julia process.

This is optimized for post-hoc cross-evaluation where a model should be loaded once,
then scored on many test datasets sequentially.

Environment variables:
  MODEL_PATH      : path to .jls model (required)
  TEST_LIST_CSV   : CSV with columns: test_run_id,test_csv (required)
  OUTPUT_CSV      : output CSV for aggregate metrics per test dataset (required)
  SLACK_BUS_ID    : optional fallback slack bus (default 65)
"""

using CSV
using DataFrames
using MLJ
using Serialization
using Statistics
using SymbolicRegression

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const INCWRAP_OPS_PATH = joinpath(@__DIR__, "incwrap_ops.jl")
if isfile(INCWRAP_OPS_PATH)
    include(INCWRAP_OPS_PATH)
end

# Ensure operators exist in Main as well.
line_current(V_send::Complex, S::Complex) = conj(S / V_send)
voltage_drop(I::Complex, Z::Complex) = I * Z

MODEL_PATH = get(ENV, "MODEL_PATH", "")
TEST_LIST_CSV = get(ENV, "TEST_LIST_CSV", "")
OUTPUT_CSV = get(ENV, "OUTPUT_CSV", "")
SLACK_BUS_ID = try parse(Int, get(ENV, "SLACK_BUS_ID", "65")) catch; 65 end

isempty(MODEL_PATH) && error("MODEL_PATH is required")
isfile(MODEL_PATH) || error("MODEL_PATH does not exist: $(MODEL_PATH)")
isempty(TEST_LIST_CSV) && error("TEST_LIST_CSV is required")
isfile(TEST_LIST_CSV) || error("TEST_LIST_CSV does not exist: $(TEST_LIST_CSV)")
isempty(OUTPUT_CSV) && error("OUTPUT_CSV is required")
mkpath(dirname(OUTPUT_CSV))

function parse_py_complex(str::AbstractString)
    s = replace(strip(str), r"\s+" => "")
    startswith(s, "(") && endswith(s, ")") && (s = s[2:end-1])
    m = match(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([+-](?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)j$", s)
    m === nothing && error("Cannot parse complex token: " * str)
    real_str, imag_str = m.captures
    return ComplexF64(parse(Float64, real_str), parse(Float64, imag_str))
end

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

rmse(a, b) = sqrt(mean((a .- b).^2))
mae(a, b) = mean(abs.(a .- b))
function r2_score(a, b)
    ss_res = sum((a .- b).^2)
    ss_tot = sum((a .- mean(a)).^2)
    return 1 - ss_res / ss_tot
end

function evaluate_one(mach, test_csv::String)
    test_df = load_complex_df(test_csv)
    voltage_cols = [c for c in names(test_df) if occursin(r"^V\d+_complex$", String(c))]
    power_cols   = [c for c in names(test_df) if occursin(r"^S\d+_complex$", String(c))]
    isempty(voltage_cols) && error("No voltage columns found in test data")
    isempty(power_cols) && error("No power columns found in test data")

    model_input_names = try Vector{String}(mach.fitresult.variable_names) catch; String[] end
    model_output_names = try Vector{String}(mach.fitresult.y_variable_names) catch; String[] end

    if isempty(model_input_names)
        voltage_bus_ids = sort(parse.(Int, match.(Ref(r"^V(\d+)_complex$"), String.(voltage_cols)) .|> x -> x.captures[1]))
        slack_bus_used = SLACK_BUS_ID in voltage_bus_ids ? SLACK_BUS_ID : voltage_bus_ids[1]
        slack_col = Symbol("V$(slack_bus_used)_complex")
        model_input_names = ["Vslack"; String.(power_cols)]
        model_output_names = String.(filter(!=(slack_col), Symbol.(voltage_cols)))
    end

    input_syms = Symbol.(model_input_names)
    target_syms = Symbol.(model_output_names)

    inputs_df = DataFrame()
    for c in input_syms
        if c == :Vslack
            voltage_bus_ids = sort(parse.(Int, match.(Ref(r"^V(\d+)_complex$"), String.(voltage_cols)) .|> x -> x.captures[1]))
            slack_bus_used = SLACK_BUS_ID in voltage_bus_ids ? SLACK_BUS_ID : voltage_bus_ids[1]
            slack_col = Symbol("V$(slack_bus_used)_complex")
            inputs_df[!, c] = test_df[!, slack_col]
        elseif String(c) in names(test_df)
            inputs_df[!, c] = test_df[!, c]
        else
            inputs_df[!, c] = fill(0.0 + 0.0im, nrow(test_df))
        end
    end

    targets_df = DataFrame()
    for c in target_syms
        if String(c) in names(test_df)
            targets_df[!, c] = test_df[!, c]
        else
            targets_df[!, c] = fill(0.0 + 0.0im, nrow(test_df))
        end
    end

    X = NamedTuple{Tuple(input_syms)}(Tuple(inputs_df[!, c] for c in input_syms))
    y_true = NamedTuple{Tuple(target_syms)}(Tuple(targets_df[!, c] for c in target_syms))
    y_pred = predict(mach, X)

    mag_rmse_vals = Float64[]
    mag_mae_vals = Float64[]
    mag_r2_vals = Float64[]
    ang_rmse_vals = Float64[]
    ang_mae_vals = Float64[]
    ang_r2_vals = Float64[]

    for c in target_syms
        actual = y_true[c]
        predicted = y_pred[c]
        act_mag = abs.(actual)
        pred_mag = abs.(predicted)
        act_ang = angle.(actual)
        pred_ang = angle.(predicted)

        push!(mag_rmse_vals, rmse(act_mag, pred_mag))
        push!(mag_mae_vals, mae(act_mag, pred_mag))
        push!(mag_r2_vals, r2_score(act_mag, pred_mag))
        push!(ang_rmse_vals, rmse(act_ang, pred_ang))
        push!(ang_mae_vals, mae(act_ang, pred_ang))
        push!(ang_r2_vals, r2_score(act_ang, pred_ang))
    end

    return (
        n_samples = nrow(test_df),
        n_targets = length(target_syms),
        mag_rmse = mean(mag_rmse_vals),
        mag_mae = mean(mag_mae_vals),
        mag_r2 = mean(mag_r2_vals),
        ang_rmse = mean(ang_rmse_vals),
        ang_mae = mean(ang_mae_vals),
        ang_r2 = mean(ang_r2_vals),
    )
end

println("Loading model once: $(MODEL_PATH)")
mach = deserialize(MODEL_PATH)
println("Model loaded. Running test list: $(TEST_LIST_CSV)")

test_list_df = CSV.read(TEST_LIST_CSV, DataFrame)
required = ["test_run_id", "test_csv"]
for c in required
    c in names(test_list_df) || error("TEST_LIST_CSV missing column: $(c)")
end

rows = DataFrame(
    test_run_id = String[],
    test_csv = String[],
    status = String[],
    error = String[],
    n_samples = Int[],
    n_targets = Int[],
    mag_rmse = Float64[],
    mag_mae = Float64[],
    mag_r2 = Float64[],
    ang_rmse = Float64[],
    ang_mae = Float64[],
    ang_r2 = Float64[],
)

for r in eachrow(test_list_df)
    test_run_id = string(r.test_run_id)
    test_csv = string(r.test_csv)
    try
        m = evaluate_one(mach, test_csv)
        push!(rows, (
            test_run_id,
            test_csv,
            "ok",
            "",
            Int(m.n_samples),
            Int(m.n_targets),
            Float64(m.mag_rmse),
            Float64(m.mag_mae),
            Float64(m.mag_r2),
            Float64(m.ang_rmse),
            Float64(m.ang_mae),
            Float64(m.ang_r2),
        ))
    catch err
        push!(rows, (
            test_run_id,
            test_csv,
            "failed",
            sprint(showerror, err),
            0,
            0,
            NaN,
            NaN,
            NaN,
            NaN,
            NaN,
            NaN,
        ))
    end
end

CSV.write(OUTPUT_CSV, rows)
ok_n = count(==("ok"), rows.status)
println("Wrote $(OUTPUT_CSV)  (ok=$(ok_n), failed=$(nrow(rows)-ok_n))")
