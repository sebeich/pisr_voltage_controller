# AR(1) prosumer power scenarios (Section IV-A).
#
#   julia --project=. data/scenarios/generate_scenarios.jl                 # offline sets (scenarios_<name>.csv)
#   julia --project=. data/scenarios/generate_scenarios.jl VARIANT=rt      # real-time set (scenarios_<name>_rt.csv)
#
# VARIANT=offline: the first 500 samples of nodes 1-3 get uniform random Q in [-50, 50] kvar
#                  (used for the offline training/test sets, Figs. 3, 7, 9).
# VARIANT=rt:      no random-Q prefix (scenarios_correlated_pv_rt.csv drives the digital twin, Figs. 8, 10, 11).
# The original files were generated without a fixed seed; the shipped CSVs are the ones used in the paper.
# Outputs go to OUT_DIR (default results/IV-A_datasets_fig3/scenarios) so the shipped data is never overwritten.
using Plots
using CSV, DataFrames, Random, Distributions

_kv = Dict(split(a, "="; limit=2)[1] => split(a, "="; limit=2)[2] for a in ARGS if occursin("=", a))
_arg(k, d) = get(_kv, k, get(ENV, k, d))
const VARIANT = _arg("VARIANT", "offline")
VARIANT in ("offline", "rt") || error("VARIANT must be offline or rt")
const SUFFIX = VARIANT == "rt" ? "_rt" : ""
const OUT_DIR = abspath(_arg("OUT_DIR", joinpath(@__DIR__, "..", "..", "results", "IV-A_datasets_fig3", "scenarios")))
isempty(_arg("SEED", "")) || Random.seed!(parse(Int, _arg("SEED", "")))

# ========== CONFIGURATION ==========

const N_NODES = 7
const DURATION_S = 3600  # 1 hour in seconds

const SCENARIOS = ["random_high_var", "correlated_pv", "mixed_overload"]
# Scaling factors for output range (adjust as needed)
const P_scale = 5.0  # e.g., set to 0.1 for smaller values
const Q_scale = 0.5

# Override: for the first X samples, assign random, uncorrelated Q values
# to the first Y nodes (columns). Values are sampled uniformly from [min, max].
# Apply after scaling so the final values match this range.
const Q_RANDOM_PREFIX_SAMPLES = VARIANT == "rt" ? 0 : 500
const Q_RANDOM_COLS = 3
const Q_RANDOM_RANGE = (-50, 50)

# Define scenario parameters
function scenario_params(scenario)
    if scenario == "random_high_var"
        return (μP=5.0, σP=3.0, μQ=0.0, σQ=0.1, corr=0.0)
    elseif scenario == "correlated_pv"
        return (μP=10.0, σP=0.8, μQ=0.0, σQ=0.1, corr=0.85)
    elseif scenario == "mixed_overload"
        return (μP=[10.0, 10.0, 3.0, 3.0, 3.0, 3.0, 3.0], σP=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                μQ=[4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0], σQ=[1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5], corr=0.5)
    else
        error("Unknown scenario: $scenario")
    end
end

# ========== GENERATION FUNCTIONS ==========


# Generate AR(1) time series for each node
function ar1_series(μ, σ, α, T)
    x = zeros(T)
    x[1] = μ + randn() * σ
    for t in 2:T
        x[t] = α * x[t-1] + (1-α) * μ + randn() * σ
    end
    return x
end

function generate_setpoints(scenario)
    params = scenario_params(scenario)
    P = zeros(DURATION_S, N_NODES)
    Q = zeros(DURATION_S, N_NODES)
    α = 0.95  # temporal correlation (AR(1) coefficient)

    if scenario == "random_high_var"
        for n in 1:N_NODES
            P[:, n] = ar1_series(params.μP, params.σP, α, DURATION_S)
            Q[:, n] = ar1_series(params.μQ, params.σQ, α, DURATION_S)
        end
    elseif scenario == "correlated_pv"
        # Generate a base AR(1) profile
        base_P = ar1_series(params.μP, params.σP, α, DURATION_S)
        base_Q = ar1_series(params.μQ, params.σQ, α, DURATION_S)
        for n in 1:N_NODES
            noise_P = ar1_series(0.0, params.σP, α, DURATION_S)
            noise_Q = ar1_series(0.0, params.σQ, α, DURATION_S)
            P[:, n] = params.corr * base_P .+ (1 - params.corr) * noise_P
            Q[:, n] = params.corr * base_Q .+ (1 - params.corr) * noise_Q
        end
    elseif scenario == "mixed_overload"
        # Node-specific AR(1) series
        for n in 1:N_NODES
            P[:, n] = ar1_series(params.μP[n], params.σP[n], α, DURATION_S)
            Q[:, n] = ar1_series(params.μQ[n], params.σQ[n], α, DURATION_S)
        end
        # Add some correlation to simulate similar behavior
        base = ar1_series(0.0, 1.0, α, DURATION_S)
        for n in 1:N_NODES
            P[:, n] = params.corr * base .+ (1 - params.corr) * P[:, n]
            Q[:, n] = params.corr * base .+ (1 - params.corr) * Q[:, n]
        end
    end

    # Apply scaling (no clamping, allow negative values)
    P = P .* P_scale
    Q = Q .* Q_scale

    # Override Q for the first samples and columns with uncorrelated random values
    n_samples = min(DURATION_S, Q_RANDOM_PREFIX_SAMPLES)
    n_cols = min(N_NODES, Q_RANDOM_COLS)
    if n_samples > 0 && n_cols > 0
        Q[1:n_samples, 1:n_cols] .= rand(Uniform(Q_RANDOM_RANGE[1], Q_RANDOM_RANGE[2]), n_samples, n_cols)
    end
    return P, Q
end

# ========== MAIN SCRIPT ==========

function main()
    mkpath(OUT_DIR)
    times = collect(0:DURATION_S-1)
    for scenario in SCENARIOS
        P, Q = generate_setpoints(scenario)
        df = DataFrame(time=times)
        for n in 1:N_NODES
            df[!, "P_node$n"] = P[:, n]
            df[!, "Q_node$n"] = Q[:, n]
        end
        out_file = joinpath(OUT_DIR, "scenarios_$(scenario)$(SUFFIX).csv")
        CSV.write(out_file, df)
        println("Scenario '$(scenario)' written to $(out_file)")

        # Plot and save P and Q for each scenario
        pltP = plot(title="Active Power (P) - $(scenario)", xlabel="Time (s)", ylabel="P in kW", legend=:topright, dpi=500)
        for n in 1:N_NODES
            plot!(pltP, times, P[:, n], label="P_node$n")
        end
        pngP = joinpath(OUT_DIR, "scenarios_$(scenario)_P$(SUFFIX).png")
        savefig(pltP, pngP)

        pltQ = plot(title="Reactive Power (Q) - $(scenario)", xlabel="Time (s)", ylabel="Q in kVAr", legend=:topright, dpi=500)
        for n in 1:N_NODES
            plot!(pltQ, times, Q[:, n], label="Q_node$n")
        end
        pngQ = joinpath(OUT_DIR, "scenarios_$(scenario)_Q$(SUFFIX).png")
        savefig(pltQ, pngQ, )
        println("Plots for scenario '$(scenario)' saved to $(pngP) and $(pngQ)")
    end
end

main()