import argparse
import glob
import json
import multiprocessing
import os

import cv2
import numpy as np
import pydicom
import tqdm
from skimage.draw import polygon2mask
from sklearn.model_selection import KFold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("output_dir", type=str)
    return parser.parse_args()


def run_single(dcm_path, result_path, img_save_dir, mask_save_dir):
    dcm = pydicom.dcmread(dcm_path)
    with open(result_path, "r") as f:
        data = json.load(f)
    frame_number = data["frameNo"]
    img = dcm.pixel_array[frame_number]
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    contour = np.array(json.loads(data["editContour"]))
    mask = polygon2mask(img.shape[:2], contour[:, ::-1]).astype(np.uint8)
    token = data["token"]
    basename = f"{token}.png"

    cv2.imwrite(os.path.join(img_save_dir, basename), img)
    cv2.imwrite(os.path.join(mask_save_dir, basename), mask)


def create_directories(output_dir):
    img_dir = os.path.join(output_dir, "img_dir")
    ann_dir = os.path.join(output_dir, "ann_dir")
    os.makedirs(os.path.join(img_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(img_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(img_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(ann_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(ann_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(ann_dir, "test"), exist_ok=True)
    return img_dir, ann_dir


def main():
    args = parse_args()
    assert not os.path.exists(args.output_dir)

    result_path_list = glob.glob(
        os.path.join(args.dataset_dir, "**/*.json"), recursive=True)
    dcm_path_list = []
    for res_path in result_path_list:
        dirname = os.path.dirname(res_path)
        filename = f"{os.path.basename(res_path).split('_')[0]}.dcm"
        dcm_path_list.append(os.path.join(dirname, filename))
    token_list = [
        os.path.splitext(os.path.basename(i))[0] for i in dcm_path_list
    ]

    img_dir, ann_dir = create_directories(args.output_dir)

    # Split into train/val/test
    test_size = int(0.1 *
                    len(token_list))  # Define test size as 10% of total data
    test_ids = token_list[:test_size]
    train_val_ids = token_list[test_size:]

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    train_indices, val_indices = next(
        kf.split(train_val_ids))  # Get new train/val split from remaining data
    train_ids = [train_val_ids[i] for i in train_indices]
    val_ids = [train_val_ids[i] for i in val_indices]

    # Prepare list of arguments for multiprocessing
    task_list = []
    for dcm_path, result_path, token in zip(dcm_path_list, result_path_list,
                                            token_list):
        if token in train_ids:
            img_save_dir = os.path.join(img_dir, "train")
            mask_save_dir = os.path.join(ann_dir, "train")
        elif token in val_ids:
            img_save_dir = os.path.join(img_dir, "val")
            mask_save_dir = os.path.join(ann_dir, "val")
        elif token in test_ids:
            img_save_dir = os.path.join(img_dir, "test")
            mask_save_dir = os.path.join(ann_dir, "test")
        else:
            continue  # Skip if token is not in any set (unlikely)

        task_list.append((dcm_path, result_path, img_save_dir, mask_save_dir))

    # Run all tasks using a single multiprocessing pool
    with multiprocessing.Pool() as pool:
        pool.starmap(run_single, tqdm.tqdm(task_list, total=len(task_list)))


if __name__ == "__main__":
    main()
