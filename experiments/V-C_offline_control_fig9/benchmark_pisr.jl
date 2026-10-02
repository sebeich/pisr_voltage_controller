# Fair speed benchmark, PISR side (Sec. V-C).
#
#   julia --project=. experiments/V-C_offline_control_fig9/benchmark_pisr.jl [OUT_JSON]
#
# Evaluates the PISR surrogate in-process, one input vector at a time, on the 100 samples of
# data/offline/test_data_complex_blockrand.csv: all target equations are evaluated and the
# maximum |V| is returned, i.e. exactly the quantity the optimizer needs per objective call.
# Counterpart: benchmark_powerflow.py (pandapower runpp, in-process, same inputs). The equations are
# also exported to the JSON so that benchmark_powerflow.py can evaluate them in plain Python as well,
# i.e. in the same language/runtime as pandapower.
using BenchmarkTools, CSV, DataFrames, JSON3, MLJ, Serialization, Statistics, SymbolicRegression

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const MODEL_PATH = get(ENV, "MODEL_PATH", joinpath(ROOT, "models", "offline_highvar", "final_model_iter1020.jls"))
const TEST_CSV = joinpath(ROOT, "data", "offline", "test_data_complex_blockrand.csv")
const OUT_JSON = length(ARGS) >= 1 ? ARGS[1] : joinpath(ROOT, "results", "V-C_offline_control_fig9", "benchmark_pisr.json")

# custom operators used in training (needed to deserialize the expression trees)
line_current(V_send::Complex, S::Complex) = conj(S / V_send)
voltage_drop(I::Complex, Z::Complex) = I * Z

function parse_py_complex(s::AbstractString)
    t = replace(strip(s), r"[()\s]" => "", "j" => "im")
    return parse(ComplexF64, t)
end

mach = deserialize(MODEL_PATH)
fr = mach.fitresult
inputs = Symbol.(fr.variable_names)
rep = report(mach)
eqs = [rep.equations[i][rep.best_idx[i]] for i in eachindex(rep.best_idx)]

df = CSV.read(TEST_CSV, DataFrame; types=String)
X = [ComplexF64[parse_py_complex(df[r, c]) for c in inputs] for r in 1:nrow(df)]

const xbuf = Matrix{ComplexF64}(undef, length(inputs), 1)
function max_abs_v(x::Vector{ComplexF64})
    xbuf[:, 1] .= x
    m = 0.0
    @inbounds for eq in eqs
        y = eval_tree_array(eq, xbuf)
        v = (y isa Tuple ? y[1] : y)[1]
        m = max(m, abs(v))
    end
    return m
end

# high-level MLJ predict on a 1-row table, for reference
function max_abs_v_mlj(x::Vector{ComplexF64})
    nt = NamedTuple{Tuple(inputs)}(Tuple([v] for v in x))
    pred = MLJ.predict(mach, nt)
    return maximum(abs(col[1]) for col in values(pred))
end

max_abs_v(X[1]); max_abs_v_mlj(X[1])          # compile
t_low = [median(@benchmark max_abs_v($x) seconds=0.2).time / 1e9 for x in X]     # median per input, in s
t_mlj = [median(@benchmark max_abs_v_mlj($x) seconds=0.2).time / 1e9 for x in X]
vmax = [max_abs_v(x) for x in X]

env = Dict(
    "cpu" => Sys.cpu_info()[1].model, "logical_cpus" => Sys.CPU_THREADS,
    "memory_gb" => round(Sys.total_memory() / 2^30, digits=1), "os" => string(Sys.KERNEL, " ", Sys.MACHINE),
    "julia" => string(VERSION), "julia_threads" => Threads.nthreads(),
    "SymbolicRegression" => string(pkgversion(SymbolicRegression)), "BenchmarkTools" => string(pkgversion(BenchmarkTools)),
    "timed_process" => "single Julia task, single-threaded evaluation, BenchmarkTools @benchmark, median per input",
)
res = Dict(
    "environment" => env,
    "model" => relpath(MODEL_PATH, ROOT), "n_inputs" => length(inputs), "n_equations" => length(eqs),
    "n_samples" => length(X), "threads" => Threads.nthreads(),
    "lowlevel_median_s" => median(t_low), "lowlevel_p05_s" => quantile(t_low, 0.05), "lowlevel_p95_s" => quantile(t_low, 0.95),
    "mlj_predict_median_s" => median(t_mlj),
    "vmax_pred" => vmax,
    "inputs" => String.(inputs),
    "equations" => [string_tree(eq) for eq in eqs],   # exported for the language-matched benchmark in Python
)
mkpath(dirname(OUT_JSON))
open(OUT_JSON, "w") do io; JSON3.write(io, res); end
println("PISR low-level evaluation (all $(length(eqs)) equations, one input): median $(round(1e6*median(t_low), digits=2)) us",
        " (p5 $(round(1e6*quantile(t_low,0.05), digits=2)), p95 $(round(1e6*quantile(t_low,0.95), digits=2)))")
println("PISR via MLJ.predict (1-row table): median $(round(1e6*median(t_mlj), digits=1)) us")
println("Saved ", OUT_JSON)
