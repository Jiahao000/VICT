# !/bin/bash

set -x

PARTITION=$1
JOB_NAME=$2
SPLIT=$3
PROMPT=$4  # ADE_train_00009574
PERTURB=$5
LEVEL=$6
TAG=$7
STEP=$8
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
PY_ARGS=${@:9}

TASK="ade20k_semseg"
SIZE=448
ROOT_PATH="output/test_time_train/painter_vit_large"
DST_DIR="${ROOT_PATH}/${TASK}_inference_${SPLIT}_${PROMPT}_size${SIZE}/${PERTURB}/${LEVEL}/${TAG}/results/${STEP}"

if ! [ -d "$DST_DIR" ]; then
   mkdir -p $DST_DIR
fi

sbatch << EOT
#!/bin/bash

#SBATCH -p ${PARTITION}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --gres gpu:${GPUS_PER_NODE}
#SBATCH -n ${GPUS}
#SBATCH --ntasks-per-node=${GPUS_PER_NODE}
#SBATCH -c ${CPUS_PER_TASK}
#SBATCH -o ${DST_DIR}/eval-%A_%a.out
#SBATCH -t 0-00:10:00

# postprocessing and eval
srun python -u eval/ade20k_semantic/ADE20kSemSegEvaluatorCustom.py \
  --pred_dir ${DST_DIR}/results --prompt ${PROMPT} \
  ${PY_ARGS}

EOT
