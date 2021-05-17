printf "======================================\n"
printf "Switching state-space model\n"
printf "======================================\n"

printf "ForneyLab (EMFVMP)\n"
for i in $(seq 10); do 
    julia ./fl-vmp.jl
done

printf "ForneyLab (SVMP)\n"
for i in $(seq 10); do 
    julia ./fl-svmp.jl
done

printf "Turing (HMC)\n"
for i in $(seq 10); do 
    julia ./hmc.jl
done

printf "Turing (NUTS)\n"
for i in $(seq 10); do 
    julia ./nuts.jl
done
