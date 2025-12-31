# !/bin/bash

set -x

PARTITION=$1
JOB_NAME=$2
SPLIT=$3
PROMPT=$4  # 000000391460
PERTURB=$5
LEVEL=$6
CKPT_FILE="painter_vit_large.pth"
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
PY_ARGS=${@:7}

SIZE=448
DIST_THR=19
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"

CKPT_PATH="models/painter_vit_large/${CKPT_FILE}"
ROOT_PATH="output/inference/painter_vit_large"
DST_DIR_SEM="${ROOT_PATH}/pano_semseg_inference_${CKPT_FILE}_${SPLIT}_${PROMPT}_size${SIZE}/baseline/${PERTURB}/${LEVEL}"
DST_DIR_INS="${ROOT_PATH}/pano_inst_inference_${CKPT_FILE}_${SPLIT}_${PROMPT}_size${SIZE}/baseline/${PERTURB}/${LEVEL}"

if ! [ -d "$DST_DIR_SEM" ]; then
   mkdir -p $DST_DIR_SEM
fi

if ! [ -d "$DST_DIR_INS" ]; then
   mkdir -p $DST_DIR_INS
fi

sbatch << EOT
#!/bin/bash

#SBATCH -p ${PARTITION}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --gres gpu:${GPUS_PER_NODE}
#SBATCH -n ${GPUS}
#SBATCH --ntasks-per-node=${GPUS_PER_NODE}
#SBATCH -c ${CPUS_PER_TASK}
#SBATCH -o ${DST_DIR_SEM}/eval-%A_%a.out
#SBATCH -t 0-02:30:00

# postprocessing and eval
srun python -u eval/coco_panoptic/COCOInstSegEvaluatorCustom.py \
  --semseg_work_dir ${DST_DIR_SEM}/results --inst_work_dir ${DST_DIR_INS}/results --ckpt_file ${CKPT_FILE} \
  --dist_thr ${DIST_THR} --prompt ${PROMPT} --input_size ${SIZE} \
  ${PY_ARGS}

srun python -u eval/coco_panoptic/COCOPanoEvaluatorCustom.py \
  --semseg_work_dir ${DST_DIR_SEM}/results --inst_work_dir ${DST_DIR_INS}/results --ckpt_file ${CKPT_FILE} \
  --dist_thr ${DIST_THR} --prompt ${PROMPT} --input_size ${SIZE} \
  ${PY_ARGS}

EOT
