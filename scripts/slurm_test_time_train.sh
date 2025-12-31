#!/bin/bash

set -x

PARTITION=$1
JOB_NAME=$2
SPLIT=$3  # train train-c
PROMPT=$4  # study_room_0005b/rgb_00094 ADE_train_00009574 000000391460 000000391460 9_9 100 100
PROMPT_IMG_PATH=$5
PROMPT_TGT_PATH=$6
PERTURB=$7
LEVEL=$8
TASK=$9  # nyuv2_depth ade20k_semseg pano_semseg pano_inst sidd derain lol
TAG=${10}
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
PY_ARGS=${@:11}  # --val_json_path --port

DATA_PATH="datasets"
SIZE=448
MODEL="painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
ROOT_PATH="output/test_time_train/painter_vit_large"
WORK_DIR="${ROOT_PATH}/${TASK}_inference_${SPLIT}_${PROMPT}_size${SIZE}/${PERTURB}/${LEVEL}/${TAG}"

if ! [ -d "$WORK_DIR" ]; then
   mkdir -p $WORK_DIR
fi

sbatch << EOT
#!/bin/bash

#SBATCH -p ${PARTITION}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --gres gpu:${GPUS_PER_NODE}
#SBATCH -n ${GPUS}
#SBATCH --ntasks-per-node=${GPUS_PER_NODE}
#SBATCH -c ${CPUS_PER_TASK}
#SBATCH -o ${WORK_DIR}/train-%A_%a.out

srun python -u main_test_time_train.py \
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
    ${PY_ARGS}

EOT
