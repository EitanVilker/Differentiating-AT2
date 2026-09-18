#!/bin/bash -l

# Set SCC project
#$  -P crem-trainees
#$ -pe omp 16
#$ -l mem_per_core=16G
#$ -t 1-2
#$ -N BestGenes
#$ -j y
#$ -o bestGenes.out
#$ -e bestGenes.err

# module load python3/3.13.8
python -V

./bestGenesBatchJobSingle.py $SGE_TASK_ID "LungMAP"