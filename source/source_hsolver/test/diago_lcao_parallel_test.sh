#!/bin/bash

np=`cat /proc/cpuinfo | grep "cpu cores" | uniq| awk '{print $NF}'`
echo "nprocs in this machine is $np"

for i in 6 3 2;do
    if [[ $i -gt $np ]];then
        continue
    fi
    echo "TEST DIAGO davidson in parallel, nprocs=$i"
    mpirun -np $i ./MODULE_HSOLVER_LCAO
    mpirun -np $i ./MODULE_HSOLVER_dav_real
    mpirun -np $i ./MODULE_HSOLVER_cg_real
    if [[ $? != 0 ]];then
        echo -e "\e[1;33m [  FAILED  ] \e[0m"\
			"execute UT with $i cores error."
        exit 1
    fi
    break    
done
