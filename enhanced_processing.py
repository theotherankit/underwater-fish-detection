import os
import shutil
import re
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def classify_images(input_dir):
    """
    Classifies images in 'train', 'test', 'valid' folders into class-named subfolders
    based on the prefix of the filename.
    Returns a list of new sorted split folder names.
    """
    splits = ['train', 'test', 'valid']
    sorted_splits = []
    for split in splits:
        src_dir = os.path.join(input_dir, split)
        dst_dir = os.path.join(input_dir, f"{split}_sorted")
        sorted_splits.append(f"{split}_sorted")
        if not os.path.exists(src_dir):
            print(f"Warning: {src_dir} does not exist, skipping.")
            continue
        os.makedirs(dst_dir, exist_ok=True)
        for filename in os.listdir(src_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                base_name = filename.split("_")[0]
                match = re.match(r"[A-Za-z]+", base_name)
                if match:
                    class_name = match.group().lower()
                    class_path = os.path.join(dst_dir, class_name)
                    os.makedirs(class_path, exist_ok=True)
                    shutil.copy2(os.path.join(src_dir, filename), os.path.join(class_path, filename))
                else:
                    print(f"Warning: Could not classify {filename}, skipping.")
    return sorted_splits

def apply_gentle_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced

def gentle_white_balance(img):
    b, g, r = cv2.split(img)
    r_avg = np.mean(r)
    g_avg = np.mean(g)
    b_avg = np.mean(b)
    k = 0.5
    avg = (r_avg + g_avg + b_avg) / 3
    r_scale = 1 + k * ((avg / r_avg) - 1) if r_avg > 0 else 1
    g_scale = 1 + k * ((avg / g_avg) - 1) if g_avg > 0 else 1
    b_scale = 1 + k * ((avg / b_avg) - 1) if b_avg > 0 else 1
    r = cv2.addWeighted(r, r_scale, 0, 0, 0)
    g = cv2.addWeighted(g, g_scale, 0, 0, 0)
    b = cv2.addWeighted(b, b_scale, 0, 0, 0)
    balanced = cv2.merge((b, g, r))
    return balanced

def enhance_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.addWeighted(s, 1.2, 0, 0, 0)
    s = np.clip(s, 0, 255).astype(np.uint8)
    merged = cv2.merge((h, s, v))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)
    return enhanced

def enhance_underwater_image(img):
    contrast_enhanced = apply_gentle_clahe(img)
    color_enhanced = enhance_color(contrast_enhanced)
    final = gentle_white_balance(color_enhanced)
    return final

def enhance_and_save(sorted_splits, input_dir, output_dir):
    """
    Enhances images in sorted split folders and saves them to output_dir with same structure.
    """
    for split in sorted_splits:
        split_input = os.path.join(input_dir, split)
        split_output = os.path.join(output_dir, split.replace('_sorted', ''))
        if not os.path.exists(split_input):
            print(f"Skipping {split_input}, does not exist.")
            continue
        for class_name in os.listdir(split_input):
            class_input = os.path.join(split_input, class_name)
            if not os.path.isdir(class_input):
                continue
            class_output = os.path.join(split_output, class_name)
            os.makedirs(class_output, exist_ok=True)
            images = [f for f in os.listdir(class_input) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Processing {len(images)} images in {split}/{class_name}")
            for img_name in tqdm(images):
                img_path = os.path.join(class_input, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to read {img_path}")
                    continue
                enhanced_img = enhance_underwater_image(img)
                output_path = os.path.join(class_output, img_name)
                cv2.imwrite(output_path, enhanced_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify and enhance images into class subfolders")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to Dataset/RawDataset")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to Dataset/EnhancedDataset")
    args = parser.parse_args()

    print("Step 1: Classifying images into class subfolders...")
    sorted_splits = classify_images(args.input_dir)
    print("Classification done.\n")

    print("Step 2: Enhancing images and saving to output directory...")
    enhance_and_save(sorted_splits, args.input_dir, args.output_dir)
    print("Enhancement complete.")
