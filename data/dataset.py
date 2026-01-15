import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
import os
from tqdm import tqdm
from .augmentations import augment_hsv, Mosaic, Mixup

class VOCDataset(Dataset):
    def __init__(self, img_list_path='./data/voc_images_list.pt', label_dir='./data/labels', img_size=640, augment=True, hyp=None):
        try:
             self.img_files = torch.load(img_list_path, weights_only=False)
        except TypeError:
             # Fallback for older pytorch versions if they don't support weights_only arg (unlikely for 2.6 err but good practice)
             self.img_files = torch.load(img_list_path)
             
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp if hyp else {'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'mosaic': 1.0, 'mixup': 0.0}
        
        # Check for cache file using hash of img_list_path or just assume consistent name
        self.cache_path = Path(img_list_path).with_suffix('.cache')
        self.labels = self._cache_labels()
        
        # Augmentations
        self.mosaic = Mosaic(img_size, p=self.hyp['mosaic'])
        self.mixup = Mixup(p=self.hyp['mixup'])

    def _cache_labels(self):
        # 1. Load cache if exists
        if self.cache_path.exists():
            print(f"Loading labels from {self.cache_path}...")
            try:
                cache = torch.load(self.cache_path, weights_only=False)
                # Simple check if cache matches current data length
                if len(cache) == len(self.img_files):
                    return cache
                print("Cache length mismatch. Re-caching...")
            except Exception as e:
                print(f"Cache load failed: {e}. Re-caching...")

        # 2. Iterate and parse
        print("Caching labels (this might take a while)...")
        labels = []
        for img_file in tqdm(self.img_files, desc="Reading Labels"):
            img_id = Path(img_file).stem
            lb_file = self.label_dir / f"{img_id}.txt"
            
            l = np.zeros((0, 5), dtype=np.float32)
            if lb_file.exists():
                with open(lb_file) as f:
                    # Filter empty lines
                    lines = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if len(lines):
                        # class x y w h
                        l = np.array(lines, dtype=np.float32)
            
            labels.append(l)
        
        # 3. Save cache
        print(f"Saving labels to {self.cache_path}...")
        torch.save(labels, self.cache_path)
        return labels

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = self._load_image(index)
        h0, w0 = img.shape[:2]  # orig hw
        labels = self.labels[index].copy()
        
        # Un-normalize if valid
        if labels.size:
            # Normalized xywh -> pixel xywh
            labels[:, 1] *= w0
            labels[:, 2] *= h0
            labels[:, 3] *= w0
            labels[:, 4] *= h0

        if self.augment:
            # Mosaic / Mixup
            # Note: Mosaic implementation expects loaded images and labels
            # We need to load others and un-normalize them too!
            
            indices = [index] + [np.random.randint(0, len(self.labels)) for _ in range(3)]
            img4 = []
            lbl4 = []
            for i in indices:
                im = self._load_image(i)
                h, w = im.shape[:2]
                lb = self.labels[i].copy()
                if lb.size:
                    lb[:, 1] *= w
                    lb[:, 2] *= h
                    lb[:, 3] *= w
                    lb[:, 4] *= h
                img4.append(im)
                lbl4.append(lb)
            
            img, labels = self.mosaic(img4[0], lbl4[0], img4, lbl4)
            
            # Mixup
            if np.random.random() < self.hyp['mixup']:
                 pass # simplified
            
            # HSV
            augment_hsv(img, self.hyp['hsv_h'], self.hyp['hsv_s'], self.hyp['hsv_v'])
            
        else:
            # Resize
            img, ratio, (dw, dh) = self.letterbox(img, self.img_size)
            # Adjust labels
            if labels.size:
                labels[:, 1] = ratio * labels[:, 1] + dw
                labels[:, 2] = ratio * labels[:, 2] + dh
                labels[:, 3] = ratio * labels[:, 3]
                labels[:, 4] = ratio * labels[:, 4]

        # Final Normalize to 0-1 relative to NEW image size? 
        # YOLO loss usually expects NORMALIZED xywh relative to output grid or image?
        # My Loss implementation:
        # `target_bboxes = targets[:, 2:] * imgsz[[1, 0, 1, 0]]`
        # It expects targets to be normalized [0,1].
        
        # So I must RE-NORMALIZE at the end of __getitem__
        h, w = img.shape[:2]
        if labels.size:
             labels[:, 1] /= w
             labels[:, 2] /= h
             labels[:, 3] /= w
             labels[:, 4] /= h

        # Convert

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        # Normalize to 0-1 float32 here as requested to speed up Trainer loop
        img = torch.from_numpy(img).float() / 255.0
        
        return img, torch.from_numpy(labels)

    def _load_image(self, index):
        path = self.img_files[index]
        
        # Robust Path Handling for Cross-Platform (Windows -> WSL)
        if not os.path.exists(path): 
            # Try to fix Windows absolute path if running on Linux/WSL
            if 'C:\\' in path or 'c:\\' in path:
                # Convert C:\Users... to /mnt/c/Users...
                path = path.replace('\\', '/').replace('C:', '/mnt/c').replace('c:', '/mnt/c')

        img = cv2.imread(path)
        if img is None:
             raise FileNotFoundError(f"Image Not Found: {path}")
        return img

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114)):
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw, dh = dw / 2, dh / 2  # divide padding into 2 sides

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (left, top)
