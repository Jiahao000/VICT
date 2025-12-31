#!/bin/bash

set -x

SPLIT=$1  # train train-c
PROMPT=$2  # study_room_0005b/rgb_00094 ADE_train_00009574 000000391460 000000391460 9_9 100 100
PROMPT_IMG_PATH=$3
PROMPT_TGT_PATH=$4
PERTURB=$5
LEVEL=$6
TASK=$7  # nyuv2_depth ade20k_semseg pano_semseg pano_inst sidd derain lol
TAG=$8
PY_ARGS=${@:9}  # --val_json_path 

DATA_PATH="datasets"
SIZE=448
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
ROOT_PATH="output/test_time_train/painter_vit_large"
WORK_DIR="${ROOT_PATH}/${TASK}_inference_${SPLIT}_${PROMPT}_size${SIZE}/${PERTURB}/${LEVEL}/${TAG}"

if ! [ -d "$WORK_DIR" ]; then
   mkdir -p $WORK_DIR
fi

torchrun --nproc_per_node=1 --master_port=29500 \
    main_test_time_train.py \
    --batch_size 1 \
    --accum_iter 1  \
    --model ${MODEL} \
    --steps_per_example 60 \
    --input_size 896 448 \
    --data_path ${DATA_PATH} \
    --output_dir ${WORK_DIR} \
    --log_dir ${WORK_DIR}/logs \
    --infer_dir ${WORK_DIR}/results \
    --finetune models/painter_vit_large/painter_vit_large.pth \
    --prompt_img_path ${PROMPT_IMG_PATH} \
    --prompt_tgt_path ${PROMPT_TGT_PATH} \
    ${PY_ARGS} \
    2>&1 | tee $WORK_DIR/train.log > /dev/null &

echo "tail -f $WORK_DIR/train.log"
