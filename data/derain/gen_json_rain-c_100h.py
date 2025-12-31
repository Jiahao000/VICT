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
    parser = argparse.ArgumentParser('Deraining corruption preparation', add_help=False)
    parser.add_argument('--split', type=str, help='dataset split', 
                        choices=['train', 'val'], required=True)
    parser.add_argument('--output_dir', type=str, help='path to output dir', 
                        default='datasets/derain-c')
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
            if args.split == 'train':
                image_dir = os.path.join(args.output_dir, corruption, str(severity), "derain/train/input")
                save_path = os.path.join(args.output_dir, corruption, str(severity), "derain", "derain-c_train.json")
            elif args.split == 'val':
                # we simply use Rain100H for loss eval
                image_dir = os.path.join(args.output_dir, corruption, str(severity), "derain/test/Rain100H/input")
                save_path = os.path.join(args.output_dir, corruption, str(severity), "derain", "derain-c_test_rain100h.json")
            else:
                raise NotImplementedError

            output_dict = []

            image_path_list = glob.glob(os.path.join(image_dir, '*.png')) + glob.glob(os.path.join(image_dir, '*.jpg'))

            for image_path in tqdm.tqdm(image_path_list):
                # image_name = os.path.basename(image_path)
                target_path = image_path.replace('input', 'target').replace(os.path.join(args.output_dir, corruption, str(severity)), 'datasets')
                assert os.path.isfile(image_path)
                assert os.path.isfile(target_path)
                pair_dict = {}
                pair_dict["image_path"] = image_path.replace(args.output_dir, 'derain-c')
                pair_dict["target_path"] = target_path.replace('datasets/', '')
                pair_dict["type"] = "derain_image2derain"
                output_dict.append(pair_dict)

            json.dump(output_dict, open(save_path, 'w'))
