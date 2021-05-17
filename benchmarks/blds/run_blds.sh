printf "======================================\n"
printf "Bayesian Linear Dynamical System\n"
printf "======================================\n"

printf "ForneyLab (EVMP)\n"
for i in $(seq 10); do 
    julia ./fl.jl
done

printf "Turing (ADVI)\n"
for i in $(seq 10); do 
    julia ./advi.jl
done

printf "Turing (NUTS)\n"
for i in $(seq 10); do 
    julia ./nuts.jl
done
