# --------------------------------------------------------
# Based on Painter, MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------

import os
import glob
import json
import tqdm
import argparse

from imagecorruptions import get_corruption_names
from imagecorruptions import corruption_dict


def get_args_parser():
    parser = argparse.ArgumentParser('COCO corruption semantic segmentation preparation', add_help=False)
    parser.add_argument('--split', type=str, help='dataset split', 
                        choices=['train2017', 'val2017'], required=True)
    parser.add_argument('--output_dir', type=str, help='path to output dir', 
                        default='datasets/coco-c')
    # corruption params
    parser.add_argument("-su", "--subset", choices=['common', 'validation', 'all', 'noise', 'blur',
                        'weather', 'digital'], help="Which subsets of corruptions should be applied")
    parser.add_argument("-c", "--corruptions", type=str, choices=corruption_dict.keys(), nargs="+",
                        help="Kind of corruptions to be applied, can be mixed with subset")
    parser.add_argument("-se", "--severity", type=int, choices=range(1, 6), nargs="*",
                        help="Severity level of corruption, if not provided all 5 levels will be applied")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args_parser()

    # make severity a list
    severity_levels = list(range(1, 6)) if args.severity is None else args.severity

    corruptions = args.corruptions if args.corruptions else []
    if args.subset:
        corruptions.extend(get_corruption_names(args.subset))
    corruptions = list(set(corruptions))  # remove duplicates
    assert len(corruptions) > 1, ValueError("No corruption types were provided")

    for corruption in tqdm.tqdm(corruptions):  # get_corruption_names(subset=subset):
        for severity in tqdm.tqdm(severity_levels):
            image_dir = os.path.join(args.output_dir, corruption, str(severity), "coco", args.split)
            panoptic_dir = 'datasets/coco/pano_sem_seg/panoptic_segm_{}_with_color'.format(args.split)
            save_path = os.path.join(args.output_dir, corruption, str(severity), "coco/pano_sem_seg", "coco-c_{}_image_panoptic_sem_seg.json".format(args.split))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            output_dict = []

            image_path_list = glob.glob(os.path.join(image_dir, '*g'))
            for image_path in tqdm.tqdm(image_path_list):
                image_name = image_path.split('/')[-1].split('.')[0]
                image_path = os.path.join(image_dir, image_name+'.jpg')
                panoptic_path = os.path.join(panoptic_dir, image_name+'.png')
                assert os.path.isfile(image_path)
                if not os.path.isfile(panoptic_path):
                    print("ignore {}".format(image_path))
                    continue
                pair_dict = {}
                pair_dict["image_path"] = image_path.replace(args.output_dir, 'coco-c')
                pair_dict["target_path"] = panoptic_path.replace('datasets/', '')
                pair_dict["type"] = "coco_image2panoptic_sem_seg"
                output_dict.append(pair_dict)

            json.dump(output_dict, open(save_path, 'w'))
