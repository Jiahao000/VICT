# !/bin/bash

set -x

PARTITION=$1
JOB_NAME=$2
SPLIT=$3
PROMPT=$4  # 100
PERTURB=$5
LEVEL=$6
CKPT_FILE="painter_vit_large.pth"
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
PY_ARGS=${@:7}

SIZE=448
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"

CKPT_PATH="models/painter_vit_large/${CKPT_FILE}"
ROOT_PATH="output/inference/painter_vit_large"
DST_DIR="${ROOT_PATH}/lol_inference_${CKPT_FILE}_${SPLIT}_${PROMPT}_size${SIZE}/baseline/${PERTURB}/${LEVEL}"

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
srun python -u eval/lol/evaluate_PSNR_SSIM.py \
  --pred_path ${DST_DIR} \
  --gt_path datasets/light_enhance/eval15/high \
  ${PY_ARGS}

EOT
