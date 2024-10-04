import argparse
import json
from pathlib import Path

import cv2
import numpy as np


class ArcadeDatasetParser:
    image_size = 512

    def __init__(self, root_dir: str | Path, mode="train"):
        if not isinstance(root_dir, Path):
            self.root_dir = Path(root_dir)
        else:
            self.root_dir = root_dir
        self.dataset_dir = self.root_dir / mode
        self.img_dir = self.dataset_dir / "images"
        self.ann_path = self.dataset_dir / "annotations" / f"{mode}.json"
        with open(self.ann_path, encoding="utf-8") as file:
            self.coco = json.load(file)
        self.img_ids = list(img_info["id"] for img_info in self.coco["images"])
        assert len(self.img_ids) == len(set(self.img_ids))
        self.max_category_id = -1
        for ann_info in self.coco["annotations"]:
            self.max_category_id = max(ann_info["category_id"],
                                       self.max_category_id)
        print(f"max_category_id: {self.max_category_id}")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        # https://github.com/cmctec/ARCADE/blob/main/useful%20scripts/create_masks.ipynb
        img_id = self.img_ids[idx]
        ann_info_list = []
        for ann in self.coco["annotations"]:
            if img_id == ann["image_id"]:
                ann_info_list.append(ann)
        for img_info in self.coco["images"]:
            if img_info["id"] == img_id:
                img_filename = img_info["file_name"]
                img_path = self.img_dir / img_filename
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        # first axis as background
        gt_mask = np.zeros(
            (self.max_category_id + 1, self.image_size, self.image_size),
            np.int32)
        cat_ids = []
        bbox_list = []
        points_list = []
        for ann_info in ann_info_list:
            points = np.array(
                [
                    ann_info["segmentation"][0][::2],
                    ann_info["segmentation"][0][1::2]
                ],
                np.int32,
            ).T
            points = points.reshape((-1, 1, 2))
            points_list.append(points.reshape(-1, 2))
            tmp = np.zeros((self.image_size, self.image_size), np.int32)
            cv2.fillPoly(tmp, [points], (1))
            gt_mask[ann_info["category_id"]] += tmp
            gt_mask[ann_info["category_id"],
                    gt_mask[ann_info["category_id"]] > 0] = 1

            cat_ids.append(ann_info["category_id"])
            bbox_list.append(np.array(ann_info["bbox"]))

        return img_filename, img, cat_ids, gt_mask, bbox_list, points_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("output_dir", type=str)
    return parser.parse_args()


# Function to merge the multi-channel mask into a single binary mask
def merge_to_binary_mask(gt_mask):
    # Sum across channels and set any positive value to 1, else 0
    binary_mask = np.sum(gt_mask, axis=0)
    binary_mask[binary_mask > 0] = 1
    return binary_mask


def main():
    args = parse_args()
    # Paths to your data
    arcade_root = Path(args.dataset_dir)  # Change to your dataset path
    mmseg_root = Path(
        args.output_dir)  # Path where you want the dataset in MMSeg format

    # Create directories
    img_dir_train = mmseg_root / "img_dir" / "train"
    img_dir_val = mmseg_root / "img_dir" / "val"
    img_dir_test = mmseg_root / "img_dir" / "test"
    ann_dir_train = mmseg_root / "ann_dir" / "train"
    ann_dir_val = mmseg_root / "ann_dir" / "val"
    ann_dir_test = mmseg_root / "ann_dir" / "test"

    img_dir_train.mkdir(parents=True, exist_ok=True)
    img_dir_val.mkdir(parents=True, exist_ok=True)
    img_dir_test.mkdir(parents=True, exist_ok=True)
    ann_dir_train.mkdir(parents=True, exist_ok=True)
    ann_dir_val.mkdir(parents=True, exist_ok=True)
    ann_dir_test.mkdir(parents=True, exist_ok=True)

    # Example of moving images to their respective folders (assuming your current dataset structure).
    for mode in ["train", "val", "test"]:
        print(f"mode: {mode}")
        dataset_parser = ArcadeDatasetParser(arcade_root, mode=mode)
        for idx in range(len(dataset_parser)):
            img_filename, img, cat_ids, gt_mask, bbox_list, points_list = dataset_parser[
                idx]

            # Save image in the MMSegmentation img_dir
            if mode == "train":
                img_out_dir = img_dir_train
                ann_out_dir = ann_dir_train
            elif mode == "val":
                img_out_dir = img_dir_val
                ann_out_dir = ann_dir_val
            else:
                img_out_dir = img_dir_test
                ann_out_dir = ann_dir_test

            img_out_path = img_out_dir / img_filename
            cv2.imwrite(str(img_out_path), img)  # Save image

            # Merge multi-channel mask to binary mask (0 or 1)
            binary_mask = merge_to_binary_mask(gt_mask)

            # Save annotation mask (here assuming we want to save the mask)
            ann_out_path = ann_out_dir / img_filename
            cv2.imwrite(str(ann_out_path), binary_mask.astype(np.uint8))

    print("Conversion complete.")


if __name__ == "__main__":
    main()