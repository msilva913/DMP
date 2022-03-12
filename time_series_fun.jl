
using PyPlot
using LaTeXStrings, KernelDensity
using Parameters, CSV, StatsBase, Statistics, Random
using NLsolve, Dierckx
using TypedTables
using DataFrames
import RDatasets
using ShiftedArrays
iris = RDatasets.dataset("datasets", "iris")
using Pandas
using QuantEcon

using TexTables

function moments(dat; lags =2, verbose=true)
    sd = DataFrames.describe(dat, :std)
    RSD = sd./DataFrames.select(sd, :y)
    sd = sd.std
    corrs = [cor(dat[:, i], dat.y) for i in 1:ncol(dat)]
    ac = zeros(ncol(dat), lags)

    for k in 2:(lags+1)
        ac[:, k-1] = [autocor(dat[:, i])[k] for i in 1:ncol(dat)]
    end

    mom = [names(dat) sd RSD corrs ac]
    mom = DataFrame(mom)
    rename!(mom, ["Variable", "SD", "RSD", "corrs", "Cor(x, x_{-1})", "Cor(x, x_{-2})"])
    Table(mom)
    if verbose
        @show round.(mom[2:end], sigdigits=3)
    end
    return mom
end


# function hamilton_filter(x; h=8)
#     ones_col = ones(size(x))
#     x_h = ShiftedArrays.lag(x, h)
#     x_h1 = ShiftedArrays.lag(x_h, 1)
#     x_h2 = ShiftedArrays.lag(x_h, 2)
#     x_h3 = ShiftedArrays.lag(x_h, 3)
#     X = [ones_col x x_h x_h1 x_h2 x_h3]
#     X = DataFrame(X)
#     # rename
#     names!(X, [:ones_col, :x, :x_h, :x_h1, :x_h2, :x_h3])
#     ols = lm(@formula(x ~ ones_col + x_h + x_h1 + x_h2 + x_h3), X)
#     return residuals(ols)
# end


function growth_filter(x)
    X = Pandas.DataFrame(x)
    X = diff(X, 1)
    X = X - mean(X)
    return X
end

# function linear_filter(x)
#     X = DataFrame()
#     X[!, :x] = x
#     T = range(1, size(X)[1], step=1)
#     X[!, :ones] = ones(size(x))
#     X[!, :time] = T
#     ols = lm(@formula(x~  time), X)
#     return residuals(ols)
# end  




