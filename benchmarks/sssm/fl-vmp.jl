using ForneyLab, Random, DelimitedFiles, DataFrames

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

g = FactorGraph()

@RV M ~ Dirichlet([100 1 1; 1 100 1; 1 1 100]) # Dirichlet prior that disfavors frequent state transitions

z = Vector{Variable}(undef, T-1) # one-hot coding
s = Vector{Variable}(undef, T-1)
x = Vector{Variable}(undef, T)
y = Vector{Variable}(undef, T)

@RV z[1] ~ Categorical([0.333, 0.333, 0.334])
@RV x[1] ~ GaussianMeanVariance(0,1)
@RV y[1] ~ GaussianMeanVariance(x[1],1)
placeholder(y[1], :y, index=1)

f(z) = z[1]*w1 + z[2]*w2 + z[3]*w3
@RV s[1] ~ Nonlinear{Sampling}(z[1],g=f)

for t = 2:T-1
    @RV x[t] ~ GaussianMeanPrecision(x[t-1],s[t-1])
    @RV y[t] ~ GaussianMeanVariance(x[t],1)
    @RV z[t] ~ Transition(z[t-1],M)
    @RV s[t] ~ Nonlinear{Sampling}(z[t],g=f)
    
    placeholder(y[t], :y, index=t)
end;

@RV x[T] ~ GaussianMeanPrecision(x[T-1],s[T-1])
@RV y[T] ~ GaussianMeanVariance(x[T],1)
placeholder(y[T], :y, index=T)

# Define posterior factorization
pfz = PosteriorFactorization()
q_M = PosteriorFactor(M, id=:M)

q_z = Vector{PosteriorFactor}(undef, T-1)
q_x = Vector{PosteriorFactor}(undef, T)
for t=1:T-1
    q_z[t] = PosteriorFactor(z[t],id=:Z_*t)
    q_x[t] = PosteriorFactor(x[t],id=:X_*t)
end
q_x[T] = PosteriorFactor(x[T],id=:X_*T)

# Compile algorithm
etime1 = @elapsed algo_mf = messagePassingAlgorithm(id=:MF, free_energy=true)

# Generate source code
etime2 = @elapsed code_mf = algorithmSourceCode(algo_mf, free_energy=true);

# Load algorithm
eval(Meta.parse(code_mf));

total_time = etime1 + etime2

# Initialize data
data = Dict(:y => y_data)
n_its = 20

# Initial posterior factors
marginals_mf = Dict{Symbol, ProbabilityDistribution}(:M => vague(Dirichlet, (3,3)))
for t = 1:T-1
    marginals_mf[:z_*t] = ProbabilityDistribution(Univariate, Categorical, p=[0.333,0.333,0.334])
    marginals_mf[:s_*t] = vague(SampleList)
    marginals_mf[:x_*t] = vague(GaussianMeanPrecision)
end
marginals_mf[:x_*T] = vague(GaussianMeanPrecision)

# Run algorithm
F_mf = Vector{Float64}(undef, n_its)
for i = 1:n_its
    etime = @elapsed stepMFM!(data, marginals_mf)
    global total_time += etime
    for k = 1:T
        etime = @elapsed step!(:MFX_*k, data, marginals_mf)
        global total_time += etime
    end
    for k = 1:T-1
        etime = @elapsed step!(:MFZ_*k, data, marginals_mf)
        global total_time += etime
    end
    F_mf[i] = freeEnergyMF(data, marginals_mf)
end

println(total_time)
println("FE: $(F_mf[end])")