# !/bin/bash

set -x

PARTITION=$1
JOB_NAME=$2
SPLIT=$3
PROMPT=$4  # study_room_0005b/rgb_00094
PERTURB=$5
LEVEL=$6
CKPT_FILE="painter_vit_large.pth"
GPUS=${GPUS:-1}
PY_ARGS=${@:7}  # --port

SIZE=448
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"

CKPT_PATH="models/painter_vit_large/${CKPT_FILE}"
ROOT_PATH="output/inference/painter_vit_large"
DST_DIR="${ROOT_PATH}/nyuv2_depth_inference_${CKPT_FILE}_${SPLIT}_${PROMPT}_size${SIZE}/baseline/${PERTURB}/${LEVEL}"

if ! [ -d "$DST_DIR" ]; then
   mkdir -p $DST_DIR
fi

# inference
torchrun --nproc_per_node=${GPUS} --master_port=29500 \
  eval/nyuv2_depth/painter_inference_depth.py \
  --model ${MODEL} --prompt ${PROMPT} --split ${SPLIT} \
  --ckpt_path ${CKPT_PATH} --input_size ${SIZE} \
  --perturbation ${PERTURB} --level ${LEVEL} \
  ${PY_ARGS} \
  2>&1 | tee $DST_DIR/inference.log > /dev/null &

echo "tail -f $DST_DIR/inference.log"
