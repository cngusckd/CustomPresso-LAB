import argparse
import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
import numpy as np
import torchvision.datasets as dsets
from tqdm import tqdm

def convert_box(size, box):
    # Convert VOC (xmin, xmax, ymin, ymax) to YOLO (x, y, w, h) normalized
    dw = 1. / size[0]
    dh = 1. / size[1]
    
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    
    return (x * dw, y * dh, w * dw, h * dh)

def convert_annotation(year, image_id, voc_path, output_label_dir, classes):
    in_file = voc_path / f'VOC{year}' / 'Annotations' / f'{image_id}.xml'
    out_file = output_label_dir / f'{image_id}.txt'
    
    if not in_file.exists():
        return
        
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(out_file, 'w') as f:
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert_box((w, h), b)
            f.write(f"{cls_id} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n")

import requests
import tarfile
import os

def download_file(url, dest_path):
    if dest_path.exists():
        print(f"File {dest_path} already exists. Skipping download.")
        return

    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Check for HTTP errors
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        print(f"Download Error: {e}")
        dest_path.unlink(missing_ok=True)

def extract_file(tar_path, dest_dir):
    print(f"Extracting {tar_path}...")
    try:
        if not tarfile.is_tarfile(tar_path):
            print("Not a valid tar file.")
            return False
            
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=dest_dir)
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./data', help='dataset path')
    opt = parser.parse_args()
    
    root_dir = Path(opt.path) # Relative path (e.g. data/)
    root_dir.mkdir(exist_ok=True, parents=True)

    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    # Reliable High-Speed Mirrors (HuggingFace)
    # These contain the full datasets (TrainVal + Test for 2007, TrainVal for 2012)
    files = [
        {
            'url': 'https://huggingface.co/datasets/HuggingFaceM4/pascal_voc/resolve/main/voc2007.tar.gz',
            'filename': 'voc2007.tar.gz',
            'extract_name': 'VOC2007'
        },
        {
            'url': 'https://huggingface.co/datasets/HuggingFaceM4/pascal_voc/resolve/main/voc2012.tar.gz',
            'filename': 'voc2012.tar.gz',
            'extract_name': 'VOC2012'
        }
    ]
    
    print("Checking/Downloading VOC Datasets (HuggingFace Mirror)...")
    
    for item in files:
        url = item['url']
        filename = item['filename']
        extract_name = item['extract_name']
        
        filepath = root_dir / filename
        extract_dir = root_dir / extract_name # e.g. data/VOC2007
        
        # Check if extracted dir exists
        if extract_dir.exists():
            print(f"{extract_dir} already exists. Skipping download/extraction.")
            continue
            
        # 1. Download
        download_file(url, filepath)
            
        # 2. Extract
        if filepath.exists():
            extract_file(filepath, root_dir)

    # Validate Structure
    # HuggingFace tarballs extracted directly to VOC2007/VOC2012 in current dir
    # So we don't have VOCdevkit parent.
    
    # Setup Destination Labels Dir
    labels_dir = root_dir / 'labels'
    labels_dir.mkdir(exist_ok=True)
    
    # We will simply reference images in place, but we need to unify the list.
    all_img_paths = []
    
    # Separate lists for proper Train/Val split
    train_sets = [('2012', 'trainval'), ('2007', 'trainval')]
    val_sets = [('2007', 'test')]
    
    def process_sets(sets, list_name):
        img_paths = []
        print(f"Processing {list_name}...")
        for year, image_set in sets:
             # Path is now direct: data/VOC2007 etc.
            voc_year_path = root_dir / f'VOC{year}'
            
            # ImageSet file
            txt_file = voc_year_path / 'ImageSets/Main' / f'{image_set}.txt'
            
            if not txt_file.exists():
                print(f"Warning: {txt_file} not found. Skipping set.")
                continue
                
            with open(txt_file) as f:
                image_ids = f.read().strip().split()
                
            for image_id in tqdm(image_ids, desc=f"VOC{year} {image_set}"):
                # Source Image
                img_path = voc_year_path / 'JPEGImages' / f'{image_id}.jpg'
                
                # Convert Annotations (Idempotent, fast if exists? No, we should check existence to speed up)
                # But convert_annotation reads xml each time.
                # To speed up "checking", we can skip convert if txt exists?
                # User asked for "using validation set" to speed up checking.
                # Let's optimize: Only convert if destination label doesn't exist?
                # Destination: labels_dir / {image_id}.txt
                label_file = labels_dir / f'{image_id}.txt'
                if not label_file.exists():
                     convert_annotation(year, image_id, root_dir, labels_dir, classes)
                
                # Add RELATIVE path to list (POSIX format for cross-platform)
                if img_path.exists():
                    img_paths.append(img_path.as_posix())
        
        save_path = root_dir / f'voc_{list_name}_list.pt'
        torch.save(img_paths, save_path)
        print(f"Saved {len(img_paths)} images to {save_path}")

    print("Generating Train List...")
    process_sets(train_sets, 'train')
    
    print("Generating Val List...")
    process_sets(val_sets, 'val')
    
    print(f"Labels checked/generated in {labels_dir}")

if __name__ == "__main__":
    main()

