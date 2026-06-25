#!/bin/bash

mol=$1
basis=$2

filename="./results/${mol}_${basis}.json"

mpirun -np 2 \
	--mca coll ^hcoll  \
        -x CUDA_VISIBLE_DEVICES=0,1 \
        python fci_from_fcidump.py \
	--fcidump '/PATH/TO/fcidump_Fe2S2_MO.txt'\
	--mol $mol \
	--basis $basis  --max_cycle 1000 --max_space 16 \
	--filename $filename \
        --incpu 0 --debugmode 0 #>results/${mol}_${basis}.log
