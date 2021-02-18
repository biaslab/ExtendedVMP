printf "======================================\n"
printf "Bayesian Linear Dynamical System\n"
printf "======================================\n"

printf "ForneyLab (EVMP)\n"
for i in $(seq 5); do 
    julia ./fl.jl
done

printf "Turing (ADVI)\n"
for i in $(seq 5); do 
    julia ./advi.jl
done

printf "Turing (NUTS)\n"
for i in $(seq 5); do 
    julia ./nuts.jl
done
