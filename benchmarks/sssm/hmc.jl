using Turing, Random, DelimitedFiles, DataFrames

using Logging
Logging.disable_logging(Logging.Error)
setprogress!(false)

Random.seed!(0);
#Generate data
T = 120

w1, w2, w3 = 0.1, 0.25, 1

x_data = [randn()]
y_data = [x_data[end]+0.1*randn()]
for t=2:25
    append!(x_data, x_data[end] + sqrt(1/w1)*randn())
    append!(y_data, x_data[end] + randn())
end
for t=26:75
    append!(x_data, x_data[end] + sqrt(1/w2)*randn())
    append!(y_data, x_data[end] + randn())
end
for t=76:T
    append!(x_data, x_data[end] + sqrt(1/w3)*randn())
    append!(y_data, x_data[end] + randn())
end

@model function SSSM(y)
    vars = [10, 4, 1]
    T = length(y)
    z = tzeros(Int,T-1)
    x = Vector(undef, T)
    M = Vector{Vector}(undef,3) # Transition matrix
    M[1] ~ Dirichlet([100,1,1])
    M[2] ~ Dirichlet([1,100,1])
    M[3] ~ Dirichlet([1,1,100])
    
    z[1] ~ Categorical(3)
    x[1] ~ Normal()
    y[1] ~ Normal(x[1],sqrt(1))
    for t = 2:T-1
        x[t] ~ Normal(x[t-1],sqrt(vars[z[t-1]]))
        y[t] ~ Normal(x[t],sqrt(1))
        z[t] ~ Categorical(vec(M[z[t-1]]))
    end
    x[T] ~ Normal(x[T-1],sqrt(vars[z[T-1]]))
    y[T] ~ Normal(x[T],sqrt(1))
end

total_time = @elapsed gibbs = Gibbs(HMC(0.2,20,:x,:M),PG(50,:z))
total_time += @elapsed chain = sample(SSSM(y_data),gibbs,100)
println(total_time)