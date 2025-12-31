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
    parser = argparse.ArgumentParser('ADE20k corruption semantic segmentation preparation', add_help=False)
    parser.add_argument('--split', type=str, help='dataset split', 
                        choices=['training', 'validation'], required=True)
    parser.add_argument('--output_dir', type=str, help='path to output dir', 
                        default='datasets/ade20k-c')
    # corruption params
    parser.add_argument("-su", "--subset", choices=['common', 'validation', 'all', 'noise', 'blur',
                        'weather', 'digital'], help="Which subsets of corruptions should be applied")
    parser.add_argument("-c", "--corruptions", type=str, choices=corruption_dict.keys(), nargs="+",
                        help="Kind of corruptions to be applied, can be mixed with subset")
    parser.add_argument("-se", "--severity", type=int, choices=range(1, 6), nargs="*",
                        help="Severity level of corruption, if not provided all 5 levels will be applied")
    return parser.parse_args()


if __name__ == '__main__':
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
            image_dir = os.path.join(args.output_dir, corruption, str(severity), "ade/ADEChallengeData2016/images", args.split)
            annos_dir = os.path.join("datasets/ade20k/annotations_with_color", args.split)
            save_path = os.path.join(args.output_dir, corruption, str(severity), "ade/ADEChallengeData2016", "ade20k-c_{}_image_semantic.json".format(args.split))

            output_dict = []

            image_path_list = glob.glob(os.path.join(image_dir, '*g'))
            for image_path in tqdm.tqdm(image_path_list):
                image_name = image_path.split('/')[-1].split('.')[0]
                image_path = os.path.join(image_dir, image_name + '.jpg')
                panoptic_path = os.path.join(annos_dir, image_name + '.png')
                assert os.path.isfile(image_path)
                assert os.path.isfile(panoptic_path)
                pair_dict = {}
                pair_dict["image_path"] = os.path.join("ade20k-c/{}/{}/ade/ADEChallengeData2016/images/{}/".format(corruption, str(severity), args.split), image_name + ".jpg")
                pair_dict["target_path"] = "ade20k/annotations_with_color/{}/".format(args.split) + image_name + ".png"
                pair_dict["type"] = "ade20k_image2semantic"
                output_dict.append(pair_dict)
            json.dump(output_dict, open(save_path, 'w'))
