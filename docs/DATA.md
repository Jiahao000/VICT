# Prepare Datasets

The datasets include [COCO](https://cocodataset.org/), [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/), [NYUDepthV2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), [Synthetic Rain Datasets](https://paperswithcode.com/dataset/synthetic-rain-datasets), [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/), and [LoL](https://daooshee.github.io/BMVC2018website/) datasets.

After processing, the datasets should look like:

```
$VICT_ROOT/datasets/
    nyu_depth_v2/
        sync/
        official_splits/
        nyu_depth_v2_labeled.mat
        nyuv2_sync_image_depth.json  # generated
        nyuv2_test_image_depth.json  # generated
    nyu_depth_v2-c/  # generated
        brightness/
            1/
                sync/
                official_splits/
                nyuv2-c_test_image_depth.json
            ...
            5/
        ...
        zoom_blur/
    ade20k/
        images/
        annotations/
        annotations_detectron2/  # generated
        annotations_with_color/  # generated
        ade20k_training_image_semantic.json  # generated
        ade20k_validation_image_semantic.json  # generated
    ade20k-c/  # generated
        brightness/
            1/ade/ADEChallengeData2016/
                images/
                ade20k-c_validation_image_semantic.json
            ...
            5/ade/ADEChallengeData2016/
        ...
        zoom_blur/
    ADEChallengeData2016/  # sim-link to $VICT_ROOT/datasets/ade20k
    coco/
        train2017/
        val2017/
        annotations/
            instances_train2017.json
            instances_val2017.json
            person_keypoints_val2017.json
            panoptic_train2017.json
            panoptic_val2017.json
            panoptic_train2017/
            panoptic_val2017/
        panoptic_semseg_val2017/  # generated
        panoptic_val2017/  # sim-link to $VICT_ROOT/datasets/coco/annotations/panoptic_val2017
        pano_sem_seg/  # generated
            panoptic_segm_train2017_with_color
            panoptic_segm_val2017_with_color
            coco_train2017_image_panoptic_sem_seg.json
            coco_val2017_image_panoptic_sem_seg.json
        pano_ca_inst/  # generated
            train_aug0/
            train_aug1/
            ...
            train_aug29/
            train_org/
            train_flip/
            val_org/
            coco_train_image_panoptic_inst.json
            coco_val_image_panoptic_inst.json
    coco-c/  # generated
        brightness/
            1/
                train2017/
                val2017/
                pano_sem_seg/
                    coco-c_val2017_image_panoptic_sem_seg.json
                pano_ca_inst/
                    train_org/
                    val_org/
                    coco-c_val_image_panoptic_inst.json
            ...
            5/
        ...
        zoom_blur/
    derain/
        train/
            input/
            target/
        test/
            Rain100H/
            Rain100L/
            Test100/
            Test1200/
            Test2800/
        derain_train.json
        derain_test.json
        derain_test_rain100h.json
    derain-c/  # generated
        brightness/
            1/derain/
                train/
                    /input
                test/
                    Rain100H/
                    Rain100L/
                    Test100/
                    Test1200/
                    Test2800/
                derain-c_test.json
                derain-c_test_rain100h.json
            ...
            5/
        ...
        zoom_blur/
    denoise/
        SIDD_Medium_Srgb/
        train/
        val/
        denoise_ssid_train.json  # generated
        denoise_ssid_val.json  # generated
    denoise-c/  # generated
        brightness/
            1/denoise/
                train/
                val/
                denoise-c_ssid_val.json
            ...
            5/denoise/
        ...
        zoom_blur/
    light_enhance/
        our485/
            low/
            high/
        eval15/
            low/
            high/
        enhance_lol_train.json  # generated
        enhance_lol_val.json  # generated
    light_enhance-c/  # generated
        brightness/
            1/light_enhance/
                our485/
                    low/
                eval15/
                    low/
                enhance-c_lol_val.json
            ...
            5/light_enhance/
        ...
        zoom_blur/
```
Please follow the following instruction to pre-process individual datasets.


## NYU Depth V2

First, download the dataset from [here](https://drive.google.com/file/d/1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP/view?usp=sharing). Please make sure to locate the downloaded file to `$VICT_ROOT/datasets/nyu_depth_v2/sync.zip`

Next, prepare [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) test set.
```bash
# get official NYU Depth V2 split file
wget -P datasets/nyu_depth_v2/ http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
# convert mat file to image files
python data/depth/extract_official_train_test_set_from_mat.py datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat data/depth/splits.mat datasets/nyu_depth_v2/official_splits/
```

Lastly, prepare json files for training and evaluation.

For clean:
```bash
python data/depth/gen_json_nyuv2_depth.py --split sync
python data/depth/gen_json_nyuv2_depth.py --split test
```
For corrupted:
```bash
python data/depth/gen_json_nyuv2-c_depth.py --split sync
python data/depth/gen_json_nyuv2-c_depth.py --split test
```

The generated json files will be saved at `$VICT_ROOT/datasets/nyu_depth_v2/` and `$VICT_ROOT/datasets/nyu_depth_v2-c/`, respectively.

## ADE20K Semantic Segmentation

First, download the dataset from the [official website](https://groups.csail.mit.edu/vision/datasets/ADE20K/), and put it in `$VICT_ROOT/datasets/`. Afterward, unzip the zip file and rename the target folder as `ade20k`. The ADE20k folder should look like:
```
ade20k/
    images/
    annotations/
```

Second, prepare annotations for training using the following command. The generated annotations will be saved at `$VICT_ROOT/datasets/ade20k/annotations_with_color/`.
```bash
python data/ade20k/gen_color_ade20k_sem.py --split training
python data/ade20k/gen_color_ade20k_sem.py --split validation
```

Third, prepare json files for training and evaluation.

For clean:
```bash
python data/ade20k/gen_json_ade20k_sem.py --split training
python data/ade20k/gen_json_ade20k_sem.py --split validation
```
For corrupted:
```bash
python data/ade20k/gen_json_ade20k-c_sem.py --split training
python data/ade20k/gen_json_ade20k-c_sem.py --split validation
```

The generated json files will be saved at `$VICT_ROOT/datasets/ade20k/` and `$VICT_ROOT/datasets/ade20k-c/`, respectively.

Lastly, to enable evaluation with detectron2, link `$VICT_ROOT/datasets/ade20k` to `$VICT_ROOT/datasets/ADEChallengeData2016` and run:
```bash
# ln -s $VICT_ROOT/datasets/ade20k datasets/ADEChallengeData2016
python data/prepare_ade20k_sem_seg.py
```

## COCO Panoptic Segmentation
Download the COCO2017 dataset and the corresponding panoptic segmentation annotation. The COCO folder should look like:
```
coco/
    train2017/
    val2017/
    annotations/
        instances_train2017.json
        instances_val2017.json
        panoptic_train2017.json
        panoptic_val2017.json
        panoptic_train2017/
        panoptic_val2017/
```

### Prepare Data for COCO Semantic Segmentation
Prepare annotations for training using the following command. The generated annotations will be saved at `$VICT_ROOT/datasets/coco/pano_sem_seg/`.
```bash
python data/coco_semseg/gen_color_coco_panoptic_segm.py --split train2017
python data/coco_semseg/gen_color_coco_panoptic_segm.py --split val2017
```

Prepare json files for training and evaluation.

For clean:
```bash
python data/coco_semseg/gen_json_coco_panoptic_segm.py --split train2017
python data/coco_semseg/gen_json_coco_panoptic_segm.py --split val2017
```
For corrupted:
```bash
python data/coco_semseg/gen_json_coco-c_panoptic_segm.py --split train2017
python data/coco_semseg/gen_json_coco-c_panoptic_segm.py --split val2017
```

The generated json files will be saved at `$VICT_ROOT/datasets/coco/pano_sem_seg/` and `$VICT_ROOT/datasets/coco-c/corruption/severity/coco/pano_sem_seg/`, respectively.

### Prepare Data for COCO Class-Agnostic Instance Segmentation 

First, pre-process the dataset using the following command, the painted ground truth will be saved to `$VICT_ROOT/datasets/coco/pano_ca_inst`. 

```bash
cd $VICT_ROOT/data/mmdet_custom

# generate training data with common data augmentation for instance segmentation, 
# note we generate 30 copies by alternating train_aug{idx} in configs/coco_panoptic_ca_inst_gen_aug.py
./tools/dist_train.sh configs/coco_panoptic_ca_inst_gen_aug.py 1
# generate training data with only horizontal flip augmentation
./tools/dist_train.sh configs/coco_panoptic_ca_inst_gen_orgflip.py 1
# generate training data w/o data augmentation
./tools/dist_train.sh configs/coco_panoptic_ca_inst_gen_org.py 1

# generate validation data (w/o data augmentation)
./tools/dist_test.sh configs/coco_panoptic_ca_inst_gen_org.py none 1 --eval segm
```

Next, prepare json files for training and evaluation.

For clean:
```bash
cd $VICT_ROOT
python data/mmdet_custom/gen_json_coco_panoptic_inst.py --split train
python data/mmdet_custom/gen_json_coco_panoptic_inst.py --split val
```
For corrupted:
```bash
cd $VICT_ROOT
python data/mmdet_custom/gen_json_coco-c_panoptic_inst.py --split train
python data/mmdet_custom/gen_json_coco-c_panoptic_inst.py --split val
```

The generated json files will be saved at `$VICT_ROOT/datasets/coco/pano_ca_inst` and `$VICT_ROOT/datasets/coco-c/corruption/severity/coco/pano_ca_inst`, respectively.

Lastly, to enable evaluation with detectron2, link `$VICT_ROOT/datasets/coco/annotations/panoptic_val2017` to `$VICT_ROOT/datasets/coco/panoptic_val2017` and run:
```bash
# ln -s $VICT_ROOT/datasets/coco/annotations/panoptic_val2017 datasets/coco/panoptic_val2017
python data/prepare_coco_semantic_annos_from_panoptic_annos.py
```

## Low-Level Vision Tasks

### Deraining
We follow [MPRNet](https://github.com/swz30/MPRNet) to prepare the data for deraining.

Download the dataset following the instructions in [MPRNet](https://github.com/swz30/MPRNet/blob/main/Deraining/Datasets/README.md), and put it in `$VICT_ROOT/datasets/derain/`. The folder should look like:
```bash
derain/
    train/
        input/
        target/
    test/
        Rain100H/
        Rain100L/
        Test100/
        Test1200/
        Test2800/
```

Next, prepare json files for training and evaluation.

For clean:
```bash
python data/derain/gen_json_rain.py --split train
python data/derain/gen_json_rain.py --split val
```
For corrupted:
```bash
python data/derain/gen_json_rain-c.py --split train
python data/derain/gen_json_rain-c.py --split val
```

The generated json files will be saved at `$VICT_ROOT/datasets/derain/` and `$VICT_ROOT/datasets/derain-c/`, respectively.

### Denoising
We follow [Uformer](https://github.com/ZhendongWang6/Uformer) to prepare the data for SIDD denoising dataset.

For training data of SIDD, you can download the SIDD-Medium dataset from the [official url](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php). For evaluation on SIDD, you can download data from [here](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/Ev832uKaw2JJhwROKqiXGfMBttyFko_zrDVzfSbFFDoi4Q?e=S3p5hQ).

Next, generate image patches for training by the following command:
```bash
python data/sidd/generate_patches_SIDD.py --src_dir datasets/denoise/SIDD_Medium_Srgb/Data --tar_dir datasets/denoise/train
```

Lastly, prepare json files for training and evaluation.

For clean:
```bash
python data/sidd/gen_json_sidd.py --split train
python data/sidd/gen_json_sidd.py --split val
```
For corrupted:
```bash
python data/sidd/gen_json_sidd-c.py --split train
python data/sidd/gen_json_sidd-c.py --split val
```

The generated json files will be saved at `$VICT_ROOT/datasets/denoise/` and `$VICT_ROOT/datasets/denoise-c/`, respectively.

### Low-Light Image Enhancement

First, download images of LOL dataset from [google drive](https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view) and put it in `$VICT_ROOT/datasets/light_enhance/`. The folder should look like:
look like:
```bash
light_enhance/
    our485/
        low/
        high/
    eval15/
        low/
        high/
```

Next, prepare json files for training and evaluation.

For clean:
```bash
python data/lol/gen_json_lol.py --split train
python data/lol/gen_json_lol.py --split val
```

For corrupted:
```bash
python data/lol/gen_json_lol-c.py --split train
python data/lol/gen_json_lol-c.py --split val
```

The generated json files will be saved at `$VICT_ROOT/datasets/light_enhance/` and `$VICT_ROOT/datasets/light_enhance-c/`, respectively.
