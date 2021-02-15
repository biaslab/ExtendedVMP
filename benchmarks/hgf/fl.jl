using ForneyLab, Random
using Plots
using DelimitedFiles, DataFrames
Random.seed!(0);

#Generate data
T = 400

vz, vy = 0.01, 0.1

z_data_0 = 0
z_data = [sin(pi/60) + sqrt(vz)*randn()]
x_data_0 = 0
x_data = [x_data_0 + sqrt(exp(z_data[1]))*randn()]
y_data = [x_data[1]+sqrt(vy)*randn()]
for t=2:T
    append!(z_data, sin(t*pi/60) + sqrt(vz)*randn())
    append!(x_data, x_data[end] + sqrt(exp(z_data[end]))*randn())
    append!(y_data, x_data[end]+sqrt(vy)*randn())
end

g = FactorGraph()

@RV m_z_t_min 
@RV v_z_t_min 
@RV z_t_min ~ GaussianMeanVariance(m_z_t_min, v_z_t_min)
model_vz = 0.1
@RV z_t ~ GaussianMeanVariance(z_t_min, model_vz)

@RV m_x_t_min
@RV v_x_t_min
@RV x_t_min ~ GaussianMeanVariance(m_x_t_min, v_x_t_min)

f(z) = 1/exp(z)
@RV wx_t ~ Nonlinear{Sampling}(z_t,g=f)
@RV x_t ~ GaussianMeanPrecision(x_t_min, wx_t)
model_vy = 0.1
@RV y_t ~ GaussianMeanVariance(x_t, model_vy)

# Placeholders for prior
placeholder(m_z_t_min, :m_z_t_min)
placeholder(v_z_t_min, :v_z_t_min)

# Placeholders for prior
placeholder(m_x_t_min, :m_x_t_min)
placeholder(v_x_t_min, :v_x_t_min)

# Placeholder for data
placeholder(y_t, :y_t);

PosteriorFactorization()

q_x = PosteriorFactor([x_t, x_t_min], id=:XMF)
q_z = PosteriorFactor(z_t, id=:ZMF)
q_z_t_min = PosteriorFactor(z_t_min, id=:ZMinMF)
etime1 = @elapsed algo_mf = messagePassingAlgorithm(id=:MF, free_energy=true);

etime2 = @elapsed source_code = algorithmSourceCode(algo_mf,free_energy=true)
eval(Meta.parse(source_code));

total_time = etime1 + etime2

# Define values for prior statistics
m_z_0, v_z_0 = 0.0, 1.0
m_x_0, v_x_0 = 0.0, 1.0

m_z = Vector{Float64}(undef, T)
v_z = Vector{Float64}(undef, T)
m_x = Vector{Float64}(undef, T)
v_x = Vector{Float64}(undef, T)

m_z_t_min, v_z_t_min = m_z_0, v_z_0
m_x_t_min, v_x_t_min = m_x_0, v_x_0
mwx_t_min, vwx_t_min = f(m_z_0), v_z_0

n_its = 10
marginals_mf = Dict()
F_mf = zeros(n_its,T)
for t = 1:T
    # Prepare data and prior statistics
    data = Dict(:y_t       => y_data[t],
                :m_z_t_min => m_z_t_min,
                :v_z_t_min => v_z_t_min,
                :m_x_t_min => m_x_t_min,
                :v_x_t_min => v_x_t_min)
    
    # Initial recognition distributions
    marginals_mf[:z_t] = ProbabilityDistribution(Univariate, GaussianMeanVariance, m=m_z_t_min, v=v_z_t_min)
    marginals_mf[:x_t_min] = ProbabilityDistribution(Univariate, GaussianMeanVariance, m=m_x_t_min, v=v_x_t_min)
    marginals_mf[:z_t_min] = ProbabilityDistribution(Univariate, GaussianMeanVariance, m=m_z_t_min, v=v_z_t_min)
    marginals_mf[:wx_t] = vague(SampleList)
    
    # Execute algorithm
    for i = 1:n_its
        etimemzmin = @elapsed stepMFZMinMF!(data, marginals_mf)
        etimemx = @elapsed stepMFXMF!(data, marginals_mf)
        etimemz = @elapsed stepMFZMF!(data, marginals_mf)
        F_mf[i,t] = freeEnergyMF(data, marginals_mf)
        global total_time += etimemzmin + etimemx + etimemz
    end

    global m_z_t_min = mean(marginals_mf[:z_t])
    global v_z_t_min = var(marginals_mf[:z_t])
    global m_x_t_min = mean(marginals_mf[:x_t])
    global v_x_t_min = var(marginals_mf[:x_t])
    global mwx_t_min = mean(marginals_mf[:wx_t])
    global vwx_t_min = var(marginals_mf[:wx_t])
    
    # Store to buffer 
    m_x[t] = m_x_t_min
    v_x[t] = v_x_t_min
    m_z[t] = m_z_t_min
    v_z[t] = v_z_t_min
end

println(total_time)