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
    parser = argparse.ArgumentParser('NYU Depth V2 corruption preparation', add_help=False)
    parser.add_argument('--split', type=str, help='dataset split', 
                        choices=['sync', 'test'], required=True)
    parser.add_argument('--output_dir', type=str, help='path to output dir', 
                        default='datasets/nyu_depth_v2-c')
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

    split2dir = {
        'sync': 'sync',
        'test': 'official_splits/test',
    }

    split_dir = split2dir[args.split]

    for corruption in tqdm.tqdm(corruptions):  # get_corruption_names(subset=subset):
        for severity in tqdm.tqdm(severity_levels):
            output_dict = []
            save_path = os.path.join(args.output_dir, corruption, str(severity), "nyuv2-c_{}_image_depth.json".format(args.split))

            src_dir = os.path.join(args.output_dir, corruption, str(severity), split_dir)
            image_path_list = glob.glob(src_dir + "/*/rgb_*.jpg")

            for image_path in tqdm.tqdm(image_path_list):
                room_name = image_path.split('/')[-2]
                frame_name = image_path.split('/')[-1].split('.')[0].split('_')[1]
                target_path = src_dir + '/' + room_name + '/sync_depth_' + frame_name + '.png'
                assert os.path.isfile(image_path)
                assert os.path.isfile(target_path)
                image_name = image_path.split('{}/'.format(args.split))[-1]
                target_name = target_path.split('{}/'.format(args.split))[-1]

                pair_dict = {}
                pair_dict["image_path"] = "nyu_depth_v2-c/{}/{}/{}/".format(corruption, str(severity), split_dir) + image_name
                # pair_dict["target_path"] = "nyu_depth_v2-c/{}/{}/{}/".format(corruption, str(severity), split_dir) + target_name
                pair_dict["target_path"] = "nyu_depth_v2/{}/".format(split_dir) + target_name
                pair_dict["type"] = "nyuv2_image2depth"
                output_dict.append(pair_dict)

            json.dump(output_dict, open(save_path, 'w'))
