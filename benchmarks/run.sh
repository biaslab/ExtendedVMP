printf "Benchmarking for Extended VMP\n"

# Bayesian LDS
cd blds
./run_blds.sh
cd ../

# Hierarchical Gaussian filter
cd hgf
./run_hgf.sh
cd ../

# Switching state-space model
cd sssm
./run_sssm.sh
cd ../