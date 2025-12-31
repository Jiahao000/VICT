# Test-Time Training

Download pre-trained ViT-Large Painter model from [Hugging Face Models](https://huggingface.co/BAAI/Painter/blob/main/painter_vit_large.pth) and put it under `$VICT_ROOT/models/painter_vit_large/`, i.e., `$VICT_ROOT/models/painter_vit_large/painter_vit_large.pth`.

Run test-time training on each of the test sets with the command:

```bash
bash scripts/dist_multi_test_time_train.sh ${PROMPT} ${PROMPT_IMG_PATH} ${PROMPT_TGT_PATH} ${TASK} ${TAG} ${PY_ARGS}
```

For example, if you want to run test-time training on ADE20K-C semantic segmentation, comment other task lines in `dist_multi_test_time_train.sh` and specify the command:

```bash
bash scripts/dist_multi_test_time_train.sh ADE_train_00009574 datasets/ade20k/images/training/ADE_train_00009574.jpg datasets/ade20k/annotations_with_color/training/ADE_train_00009574.png ade20k_semseg ttt_encoder_lr1e-6 --verbose --single_crop --finetune_mode encoder --lr 1e-6
```

Alternatively, if you run `VICT` on a cluster managed with [slurm](https://slurm.schedmd.com/):

```bash
bash scripts/slurm_multi_test_time_train.sh ${PARTITION} ${JOB_NAME} ${PROMPT} ${PROMPT_IMG_PATH} ${PROMPT_TGT_PATH} ${TASK} ${TAG} ${PY_ARGS}
```

For example, if you want to run test-time training on ADE20K-C semantic segmentation, comment other task lines in `slurm_multi_test_time_train.sh` and specify the command:

```bash
bash scripts/slurm_multi_test_time_train.sh Dummy Vict_job ADE_train_00009574 datasets/ade20k/images/training/ADE_train_00009574.jpg datasets/ade20k/annotations_with_color/training/ADE_train_00009574.png ade20k_semseg ttt_encoder_lr1e-6 --verbose --single_crop --finetune_mode encoder --lr 1e-6
```

Note: for reference, the task options and prompt examples are listed after `TASK` and `PROMPT` in `dist_multi_test_time_train.sh`/`slurm_multi_test_time_train.sh`, respectively.
