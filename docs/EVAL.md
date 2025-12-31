# Evaluation Instructions

## NYU Depth V2

To evaluate VICT on NYUv2-C depth estimation, run:
```bash
bash eval/nyuv2_depth/slurm_multi_eval_vict.sh ${PARTITION} ${JOB_NAME} ${PROMPT} ${TAG}
```

For example:

```bash
bash eval/nyuv2_depth/slurm_multi_eval_vict.sh Dummy Vict_job study_room_0005b/rgb_00094 ttt_encoder_lr1e-6
```

To evaluate Painter baseline on NYUv2-C depth estimation, first run:
```bash
bash eval/nyuv2_depth/slurm_multi_inference.sh Dummy Vict_job study_room_0005b/rgb_00094
```
Then run:
```bash
bash eval/nyuv2_depth/slurm_multi_eval.sh Dummy Vict_job study_room_0005b/rgb_00094
```

## ADE20K Semantic Segmentation

To evaluate VICT on ADE20K-C semantic segmentation, run:
```bash
bash eval/ade20k_semantic/slurm_multi_eval_vict.sh ${PARTITION} ${JOB_NAME} ${PROMPT} ${TAG}
```

For example:

```bash
bash eval/ade20k_semantic/slurm_multi_eval_vict.sh Dummy Vict_job ADE_train_00009574 ttt_encoder_lr1e-6
```

To evaluate Painter baseline on ADE20K-C semantic segmentation, first run:
```bash
bash eval/ade20k_semantic/slurm_multi_inference.sh Dummy Vict_job ADE_train_00009574
```
Then run:
```bash
bash eval/ade20k_semantic/slurm_multi_eval.sh Dummy Vict_job ADE_train_00009574
```

## COCO Panoptic Segmentation

To evaluate VICT on COCO-C panoptic segmentation, run:
```bash
bash eval/coco_panoptic/slurm_multi_eval_vict.sh ${PARTITION} ${JOB_NAME} ${PROMPT} ${TAG}
```

For example:

```bash
bash eval/coco_panoptic/slurm_multi_eval_vict.sh Dummy Vict_job 000000391460 ttt_encoder_lr1e-6
```

To evaluate Painter baseline on COCO-C panoptic segmentation, first run:
```bash
bash eval/coco_panoptic/slurm_multi_inference.sh Dummy Vict_job 000000391460
```
Then run:
```bash
bash eval/coco_panoptic/slurm_multi_eval.sh Dummy Vict_job 000000391460
```

## Low-Level Vision Tasks

### Deraining

To evaluate VICT on Rain-C deraining, run:
```bash
bash eval/derain/slurm_multi_eval_vict.sh ${PARTITION} ${JOB_NAME} ${PROMPT} ${TAG}
```

For example:

```bash
bash eval/derain/slurm_multi_eval_vict.sh Dummy Vict_job 100 ttt_encoder_lr1e-6
```

To evaluate Painter baseline on Rain-C deraining, first run:
```bash
bash eval/derain/slurm_multi_inference.sh Dummy Vict_job 100
```
Then run:
```bash
bash eval/derain/slurm_multi_eval.sh Dummy Vict_job 100
```

### Denoising

To evaluate VICT on SIDD-C denoising, run:
```bash
bash eval/sidd/slurm_multi_eval_vict.sh ${PARTITION} ${JOB_NAME} ${PROMPT} ${TAG}
```

For example:

```bash
bash eval/sidd/slurm_multi_eval_vict.sh Dummy Vict_job 9_9 ttt_encoder_lr1e-6
```

To evaluate Painter baseline on SIDD-C denoising, first run:
```bash
bash eval/sidd/slurm_multi_inference.sh Dummy Vict_job 9_9
```
Then run:
```bash
bash eval/sidd/slurm_multi_eval.sh Dummy Vict_job 9_9
```

### Low-Light Image Enhancement

To evaluate VICT on LoL-C image enhancement, run:
```bash
bash eval/lol/slurm_multi_eval_vict.sh ${PARTITION} ${JOB_NAME} ${PROMPT} ${TAG}
```

For example:

```bash
bash eval/lol/slurm_multi_eval_vict.sh Dummy Vict_job 100 ttt_encoder_lr1e-6
```

To evaluate Painter baseline on LoL-C image enhancement, first run:
```bash
bash eval/lol/slurm_multi_inference.sh Dummy Vict_job 100
```
Then run:
```bash
bash eval/lol/slurm_multi_eval.sh Dummy Vict_job 100
```
