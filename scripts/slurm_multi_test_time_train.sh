#!/bin/bash

set -x

PARTITION=$1
JOB_NAME=$2
PROMPT=$3  # study_room_0005b/rgb_00094 ADE_train_00009574 000000391460 000000391460 9_9 100 100
PROMPT_IMG_PATH=$4
PROMPT_TGT_PATH=$5
TASK=$6  # nyuv2_depth ade20k_semseg pano_semseg pano_inst sidd derain lol
TAG=$7
PY_ARGS=${@:8}
i=29500

for LEVEL in 1 2 3 4 5
do
    for SPLIT in train train-c
    do
        for PERTURB in brightness contrast defocus_blur elastic_transform fog frost gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate shot_noise snow zoom_blur
        do
            i=$((i+1))
            if [ $SPLIT == "train" ]; then  # 0-shot
                # bash scripts/slurm_test_time_train.sh ${PARTITION} ${JOB_NAME} ${SPLIT} ${PROMPT} ${PROMPT_IMG_PATH} ${PROMPT_TGT_PATH} ${PERTURB} ${LEVEL} ${TASK} ${TAG} --val_json_path datasets/nyu_depth_v2-c/${PERTURB}/${LEVEL}/nyuv2-c_test_image_depth.json --port ${i} ${PY_ARGS}
                bash scripts/slurm_test_time_train.sh ${PARTITION} ${JOB_NAME} ${SPLIT} ${PROMPT} ${PROMPT_IMG_PATH} ${PROMPT_TGT_PATH} ${PERTURB} ${LEVEL} ${TASK} ${TAG} --val_json_path datasets/ade20k-c/${PERTURB}/${LEVEL}/ade/ADEChallengeData2016/ade20k-c_validation_image_semantic.json --port ${i} ${PY_ARGS}
                # bash scripts/slurm_test_time_train.sh ${PARTITION} ${JOB_NAME} ${SPLIT} ${PROMPT} ${PROMPT_IMG_PATH} ${PROMPT_TGT_PATH} ${PERTURB} ${LEVEL} ${TASK} ${TAG} --val_json_path datasets/coco-c/${PERTURB}/${LEVEL}/coco/pano_sem_seg/coco-c_val2017_image_panoptic_sem_seg.json --port ${i} ${PY_ARGS}
                # bash scripts/slurm_test_time_train.sh ${PARTITION} ${JOB_NAME} ${SPLIT} ${PROMPT} ${PROMPT_IMG_PATH} ${PROMPT_TGT_PATH} ${PERTURB} ${LEVEL} ${TASK} ${TAG} --val_json_path datasets/coco-c/${PERTURB}/${LEVEL}/coco/pano_sem_seg/coco-c_val2017_image_panoptic_sem_seg.json --port ${i} ${PY_ARGS}
                # bash scripts/slurm_test_time_train.sh ${PARTITION} ${JOB_NAME} ${SPLIT} ${PROMPT} ${PROMPT_IMG_PATH} ${PROMPT_TGT_PATH} ${PERTURB} ${LEVEL} ${TASK} ${TAG} --val_json_path datasets/denoise-c/${PERTURB}/${LEVEL}/denoise/denoise-c_ssid_val.json --port ${i} ${PY_ARGS}
                # bash scripts/slurm_test_time_train.sh ${PARTITION} ${JOB_NAME} ${SPLIT} ${PROMPT} ${PROMPT_IMG_PATH} ${PROMPT_TGT_PATH} ${PERTURB} ${LEVEL} ${TASK} ${TAG} --val_json_path datasets/derain-c/${PERTURB}/${LEVEL}/derain/derain-c_test.json --port ${i} ${PY_ARGS}
                # bash scripts/slurm_test_time_train.sh ${PARTITION} ${JOB_NAME} ${SPLIT} ${PROMPT} ${PROMPT_IMG_PATH} ${PROMPT_TGT_PATH} ${PERTURB} ${LEVEL} ${TASK} ${TAG} --val_json_path datasets/light_enhance-c/${PERTURB}/${LEVEL}/light_enhance/enhance-c_lol_val.json --port ${i} ${PY_ARGS}
            else  # 1-shot
                # bash scripts/slurm_test_time_train.sh ${PARTITION} ${JOB_NAME} ${SPLIT} ${PROMPT} datasets/nyu_depth_v2-c/${PERTURB}/${LEVEL}/sync/${PROMPT}.jpg ${PROMPT_TGT_PATH} ${PERTURB} ${LEVEL} ${TASK} ${TAG} --val_json_path datasets/nyu_depth_v2-c/${PERTURB}/${LEVEL}/nyuv2-c_test_image_depth.json --port ${i} ${PY_ARGS}
                bash scripts/slurm_test_time_train.sh ${PARTITION} ${JOB_NAME} ${SPLIT} ${PROMPT} datasets/ade20k-c/${PERTURB}/${LEVEL}/ade/ADEChallengeData2016/images/training/${PROMPT}.jpg ${PROMPT_TGT_PATH} ${PERTURB} ${LEVEL} ${TASK} ${TAG} --val_json_path datasets/ade20k-c/${PERTURB}/${LEVEL}/ade/ADEChallengeData2016/ade20k-c_validation_image_semantic.json --port ${i} ${PY_ARGS}
                # bash scripts/slurm_test_time_train.sh ${PARTITION} ${JOB_NAME} ${SPLIT} ${PROMPT} datasets/coco-c/${PERTURB}/${LEVEL}/coco/train2017/${PROMPT}.jpg ${PROMPT_TGT_PATH} ${PERTURB} ${LEVEL} ${TASK} ${TAG} --val_json_path datasets/coco-c/${PERTURB}/${LEVEL}/coco/pano_sem_seg/coco-c_val2017_image_panoptic_sem_seg.json --port ${i} ${PY_ARGS}
                # bash scripts/slurm_test_time_train.sh ${PARTITION} ${JOB_NAME} ${SPLIT} ${PROMPT} datasets/coco-c/${PERTURB}/${LEVEL}/coco/pano_ca_inst/train_org/${PROMPT}_image_train_org.png ${PROMPT_TGT_PATH} ${PERTURB} ${LEVEL} ${TASK} ${TAG} --val_json_path datasets/coco-c/${PERTURB}/${LEVEL}/coco/pano_sem_seg/coco-c_val2017_image_panoptic_sem_seg.json --port ${i} ${PY_ARGS}
                # bash scripts/slurm_test_time_train.sh ${PARTITION} ${JOB_NAME} ${SPLIT} ${PROMPT} datasets/denoise-c/${PERTURB}/${LEVEL}/denoise/train/input/${PROMPT}.png ${PROMPT_TGT_PATH} ${PERTURB} ${LEVEL} ${TASK} ${TAG} --val_json_path datasets/denoise-c/${PERTURB}/${LEVEL}/denoise/denoise-c_ssid_val.json --port ${i} ${PY_ARGS}
                # bash scripts/slurm_test_time_train.sh ${PARTITION} ${JOB_NAME} ${SPLIT} ${PROMPT} datasets/derain-c/${PERTURB}/${LEVEL}/derain/train/input/${PROMPT}.jpg ${PROMPT_TGT_PATH} ${PERTURB} ${LEVEL} ${TASK} ${TAG} --val_json_path datasets/derain-c/${PERTURB}/${LEVEL}/derain/derain-c_test.json --port ${i} ${PY_ARGS}
                # bash scripts/slurm_test_time_train.sh ${PARTITION} ${JOB_NAME} ${SPLIT} ${PROMPT} datasets/light_enhance-c/${PERTURB}/${LEVEL}/light_enhance/our485/low/${PROMPT}.png ${PROMPT_TGT_PATH} ${PERTURB} ${LEVEL} ${TASK} ${TAG} --val_json_path datasets/light_enhance-c/${PERTURB}/${LEVEL}/light_enhance/enhance-c_lol_val.json --port ${i} ${PY_ARGS}
            fi
        done
    done
done
