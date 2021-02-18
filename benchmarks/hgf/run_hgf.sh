printf "======================================\n"
printf "Hierarchical Gaussian filter\n"
printf "======================================\n"

printf "ForneyLab (EVMP)\n"
for i in $(seq 5); do 
    julia ./fl.jl
done

printf "Turing (ADVI)\n"
for i in $(seq 5); do 
    julia ./advi.jl
done
