#!/bin/bash

PARTITION=$1
JOB_NAME=$2
SPLIT=$3
PROMPT=$4  # 000000000165_box0
PERTURB=$5
LEVEL=$6
TAG=$7
STEP=$8
CONFIG="configs/coco-c_256x192_test_offline.py"
CKPT_FILE="painter_vit_large.pth"
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
PY_ARGS=${@:9}

CHECKPOINT=none
SIZE=448

ROOT_PATH="output/test_time_train/painter_vit_large"
DST_DIR="${ROOT_PATH}/coco_pose_inference_${SPLIT}_${PROMPT}_size${SIZE}/${PERTURB}/${LEVEL}/${TAG}/results/${STEP}"

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
#SBATCH -t 0-06:00:00

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun python -u tools/test.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" --eval mAP \
  --cfg-options split=${SPLIT} prompt=${PROMPT} input_size=${SIZE} perturbation=${PERTURB} level=${LEVEL} tag=${TAG} step=${STEP} \
  ${PY_ARGS}

EOT
