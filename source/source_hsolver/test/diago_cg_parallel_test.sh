#!/bin/bash

np=`cat /proc/cpuinfo | grep "cpu cores" | uniq| awk '{print $NF}'`
echo "nprocs in this machine is $np"

for i in 6 3 2;do
    if [[ $i -gt $np ]];then
        continue
    fi
    echo "TEST DIAGO CG in parallel, nprocs=$i"
    OMP_NUM_THREADS=1 mpirun -np $i ./MODULE_HSOLVER_cg
    e1=$?
    OMP_NUM_THREADS=1 mpirun -np $i ./MODULE_HSOLVER_cg_float
    e2=$?
    if [[ e1 -ne 0 || e2 -ne 0 ]];then
        echo -e "\e[1;33m [  FAILED  ] \e[0m"\
			"execute UT with $i cores error."
        exit 1
    fi
    break    
done
