#!/bin/bash

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
PY_ARGS=${@:4}

sbatch << EOT
#!/bin/bash

#SBATCH -p ${PARTITION}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --gres gpu:${GPUS_PER_NODE}
#SBATCH -n ${GPUS}
#SBATCH --ntasks-per-node=${GPUS_PER_NODE}
#SBATCH -c ${CPUS_PER_TASK}
#SBATCH -o logs/train-%A_%a.out
#SBATCH -t 0-12:00:00

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}

EOT