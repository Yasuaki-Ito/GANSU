#!/bin/bash

mol=$1
basis=$2

filename="./logs/${mol}_${basis}.json"

#export CUDA_LAUNCH_BLOCKING=1
mpirun -np 2 \
	--mca coll ^hcoll  \
        -x CUDA_VISIBLE_DEVICES=0,1 \
        python fci_from_dumpfile.py \
	--fcidump './share/fcidump_Fe2S2_MO.txt'\
	--mol $mol \
	--basis $basis  --max_cycle 1000 --max_space 16 \
	--filename $filename \
        --incpu 0 --debugmode 2 >logs/${mol}_${basis}.log
