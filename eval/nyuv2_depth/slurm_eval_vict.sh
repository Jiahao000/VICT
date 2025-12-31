# !/bin/bash

set -x

PARTITION=$1
JOB_NAME=$2
SPLIT=$3
PROMPT=$4  # study_room_0005b/rgb_00094
PERTURB=$5
LEVEL=$6
TAG=$7
STEP=$8
CKPT_FILE="painter_vit_large.pth"
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
PY_ARGS=${@:9}

SIZE=448
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"

CKPT_PATH="models/painter_vit_large/${CKPT_FILE}"
ROOT_PATH="output/test_time_train/painter_vit_large"
DST_DIR="${ROOT_PATH}/nyuv2_depth_inference_${SPLIT}_${PROMPT}_size${SIZE}/${PERTURB}/${LEVEL}/${TAG}/results/${STEP}"

if ! [ -d "$DST_DIR" ]; then
   mkdir -p $DST_DIR
fi

sbatch << EOT
#!/bin/bash

#SBATCH -p ${PARTITION}
#SBATCH --job-name=${JOB_NAME}
#SBATCH -c ${CPUS_PER_TASK}
#SBATCH -o ${DST_DIR}/eval-%A_%a.out
#SBATCH -t 0-00:10:00

# postprocessing and eval
srun python -u eval/nyuv2_depth/eval_with_pngs.py \
  --pred_path ${DST_DIR}/results \
  --gt_path datasets/nyu_depth_v2/official_splits/test/ \
  --dataset nyu --min_depth_eval 1e-3 --max_depth_eval 10 --eigen_crop \
  ${PY_ARGS}

EOT
