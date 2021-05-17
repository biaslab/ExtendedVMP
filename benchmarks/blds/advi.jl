using Turing, LinearAlgebra, Distributions, AdvancedVI, DelimitedFiles, DataFrames

using Logging
Logging.disable_logging(Logging.Error)
setprogress!(false)

const DUMP_VALUES = false

#Generate data
using Random
import Distributions: pdf, MvNormal, rand

Random.seed!(1) # Set random seed

T = 40 # Number of timepoints

# Generative parameters
mu_0 = [8.0, 1.0] # Prior mean
V_0 = diagm(0=>ones(2)) # Prior covariance; diageye ensures an identity matrix of Diagonal type
F = [1.0 0.2; 
     -0.5 0.8] # Process matrix
H = [1.0 0.0; 
     0.0 1.0] # Observation matrix
Q = 1e-2*diagm(0=>ones(2)) # Process noise covariance
R = 1e-1*diagm(0=>ones(2)) # Observation noise covariance

# Data
x_hat = Vector{Vector{Float64}}(undef, T)
y_hat = Vector{Vector{Float64}}(undef, T)
prior_x = MvNormal(mu_0, V_0)
process_noise_dist = MvNormal(zeros(2), Q)
obs_noise_dist = MvNormal(zeros(2), R)

x_hat[1] = rand(prior_x)
y_hat[1] = H*x_hat[1] + rand(obs_noise_dist)
for t = 2:T
    x_hat[t] = F*x_hat[t-1] + rand(process_noise_dist) # Execute process
    y_hat[t] = H*x_hat[t] + rand(obs_noise_dist) # Draw observation
end

@model function BLDS(y)
    # Generative parameters
    mu_0 = [8.0, 1.0] # Prior mean
    V_0 = diagm(0=>ones(2)) # Prior covariance; diageye ensures an identity matrix of Diagonal type
    F = [1.0 0.2; 
         -0.5 0.8] # Process matrix
    H = [1.0 0.0; 
         0.0 1.0] # Observation matrix
    Q = 1e-2*diagm(0=>ones(2)) # Process noise covariance
    R = 1e-1*diagm(0=>ones(2)) # Observation noise covariance
    T = 40 # Number of timepoints
    
    # Priors
    a ~ MvNormal(zeros(4),diagm(0=>ones(4)))
    x = Vector{Vector}(undef,T)
    x[1] ~ MvNormal(mu_0, V_0)
    
    y[1] ~ MvNormal(H*x[1], R) # Observation model
    
    for t = 2:T
        x[t] ~ MvNormal(reshape(a,(2,2))*x[t-1], Q) # Process model
        y[t] ~ MvNormal(H*x[t], R) # Observation model
    end
end

etime1 = @elapsed advi = ADVI(10, 100);
etime2 = @elapsed q = vi(BLDS(y_hat), advi);
total_time = etime1 + etime2

n_its = 10
F_mf = Vector{Float64}(undef, n_its)
F_x = Vector{Float64}(undef, n_its)

F_mf[1] = - AdvancedVI.elbo(advi, q, BLDS(y_hat), 1000);
F_x[1] = total_time

for i=2:n_its
    etime = @elapsed global q = vi(BLDS(y_hat), advi, q);
    global total_time += etime
    F_mf[i] = - AdvancedVI.elbo(advi, q, BLDS(y_hat), 1000);
    F_x[i] = total_time
end

println(total_time)
println("FE: $(F_mf[end])")

if DUMP_VALUES
    writedlm("./values/ADVIFETime.txt", F_x)
    writedlm("./values/ADVIFreeEnergy.txt", F_mf)
end