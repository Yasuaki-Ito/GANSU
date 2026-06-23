#!/bin/bash

mol=$1
basis=$2

filename="./logs/${mol}_${basis}.json"


mpirun -np 2 \
	--mca coll ^hcoll  \
        -x CUDA_VISIBLE_DEVICES=0,1 \
        -x NCCL_P2P_LEVEL=NVL \
        python fci.py --mol $mol \
	--basis $basis  --max_cycle 100 --max_space 12 \
	--filename $filename --chunksize 256 \
        --incpu 0 --debugmode 2 >logs/${mol}_${basis}.log
