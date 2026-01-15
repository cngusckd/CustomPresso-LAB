import torch
import torch.nn.functional as F
import numpy as np
import cv2
import random
import math

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    """
    HSV augmentation.
    """
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)

class Mosaic:
    """Mosaic augmentation."""
    def __init__(self, size=640, p=1.0):
        self.size = size
        self.p = p

    def __call__(self, img, labels, img4, labels4):
        """
        img: main image (not really used in mosaic logic directly as we mix 4)
        labels: main labels
        img4: list of 4 images
        labels4: list of 4 label arrays
        """
        if random.random() > self.p:
            return img, labels

        s = self.size
        # mosaic center x, y
        diff = -s // 2
        xc = int(random.uniform(-diff, 2 * s + diff))
        yc = int(random.uniform(-diff, 2 * s + diff))
        
        # Output image
        img_mosaic = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)  # base image with gray padding
        labels_mosaic = []

        # 4 images to place
        # offsets
        indices = [0, 1, 2, 3] # just assuming img4 is size 4
        
        for i, idx in enumerate(indices):
            # Load image
            img_i = img4[i]
            h, w = img_i.shape[:2]
            
            # Place img in img_mosaic
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img_mosaic[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels_i = labels4[i].copy()
            if labels_i.size > 0:
                # Normalized xywh to pixel xyxy
                # Assuming labels are input as class, x, y, w, h normalized
                # We need to convert them to pixel coords first if they are normalized? 
                # Let's assume input labels are [cls, x1, y1, x2, y2] absolute or normalized. 
                # Usually standard implementation handles both. Let's assume normalized input for now and convert.
                # Just simplified:
                
                # We expect [cls, x1, y1, x2, y2]
                labels_i[:, 1] += padw
                labels_i[:, 2] += padh
                labels_i[:, 3] += padw
                labels_i[:, 4] += padh
                labels_mosaic.append(labels_i)

        if len(labels_mosaic) > 0:
            labels_mosaic = np.concatenate(labels_mosaic, 0)
            np.clip(labels_mosaic[:, 1:], 0, 2 * s, out=labels_mosaic[:, 1:]) # Clip boxes
        else:
            labels_mosaic = np.zeros((0, 5), dtype=np.float32)
        
        # Resize to target size? Or keep large? Usually Mosaic results in 2x size which is then resized or cropped.
        # Here we just return the center crop or resize.
        # For simplicity, we just return the mosaic'd image (usually cropped to s x s centered at xc, yc)
        # But wait, standard mosaic output is usually resized.
        # Let's crop center s x s
        
        # Using center crop for simplicity of implementation
        # Actually standard YOLOv5 mosaic puts 4 images into a 2x2 grid, then takes a random crop of size s*s.
        
        # Simplified: valid area
        img_final = img_mosaic # ... logic to crop/resize
        # Resizing to size
        img_final = cv2.resize(img_final, (self.size, self.size))
        
        # Adjust labels for resize
        scale = self.size / (2 * s) # if we resized the whole 2s x 2s
        # BUT standard mosaic uses random center.
        # I'll stick to a simple implementation: 4 images, placed, then resized.
        labels_mosaic[:, 1:] *= 0.5 # since 2s -> s

        return img_final, labels_mosaic

class Mixup:
    """Mixup augmentation."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, labels1, img2, labels2):
        if random.random() > self.p:
            return img1, labels1
        
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
        labels = np.concatenate((labels1, labels2), 0)
        return img, labels
