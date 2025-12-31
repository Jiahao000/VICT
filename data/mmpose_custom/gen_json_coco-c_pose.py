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
    parser = argparse.ArgumentParser('COCO corruption pose estimation preparation', add_help=False)
    parser.add_argument('--split', type=str, help='dataset split', 
                        choices=['train', 'val'], required=True)
    parser.add_argument('--output_dir', type=str, help='path to output dir', 
                        default='datasets/coco-c_pose')
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

    split = args.split

    if split == "train":
        aug_list = [
            "_aug0", "_aug1", "_aug2", "_aug3", "_aug4",
            "_aug5", "_aug6", "_aug7", "_aug8", "_aug9",
            "_aug10", "_aug11", "_aug12", "_aug13", "_aug14",
            "_aug15", "_aug16", "_aug17", "_aug18", "_aug19",
        ]
    elif split == "val":
        aug_list = ["", "_flip"]
    else:
        raise NotImplementedError

    for corruption in tqdm.tqdm(corruptions):  # get_corruption_names(subset=subset):
        for severity in tqdm.tqdm(severity_levels):
            save_path = os.path.join(args.output_dir, corruption, str(severity), "coco_pose", "coco-c_pose_256x192_{}.json".format(split))
            # print(save_path)

            output_dict = []

            for aug_idx in aug_list:
                image_dir = os.path.join(args.output_dir, corruption, str(severity), "coco_pose/data_pair/{}_256x192{}".format(split, aug_idx))
                # print(aug_idx, image_dir)
                image_path_list = glob.glob(os.path.join(image_dir, '*image.png'))

                for image_path in tqdm.tqdm(image_path_list):
                    label_path = image_path.replace("image.png", "label.png").replace(os.path.join(args.output_dir, corruption, str(severity)), 'datasets')
                    assert label_path != image_path
                    assert os.path.isfile(image_path)
                    if not os.path.isfile(label_path):
                        print("ignoring {}".format(label_path))
                        continue
                    pair_dict = {}
                    pair_dict["image_path"] = image_path.replace(args.output_dir, 'coco-c_pose')
                    pair_dict["target_path"] = label_path.replace('datasets/', '')
                    pair_dict["type"] = "coco_image2pose"
                    output_dict.append(pair_dict)

            json.dump(output_dict, open(save_path, 'w'))
