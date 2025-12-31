# !/bin/bash

set -x

PARTITION=$1
JOB_NAME=$2
PROMPT=$3  # ADE_train_00009574
PY_ARGS=${@:4}
i=29500


for LEVEL in 1 2 3 4 5
do
    for SPLIT in train train-c
    do
        for PERTURB in brightness contrast defocus_blur elastic_transform fog frost gaussian_noise glass_blur impulse_noise jpeg_compression motion_blur pixelate shot_noise snow zoom_blur
        do
            i=$((i+1))
            bash eval/ade20k_semantic/slurm_inference.sh ${PARTITION} ${JOB_NAME} ${SPLIT} ${PROMPT} ${PERTURB} ${LEVEL} --port ${i} ${PY_ARGS}
        done
    done
done
