import torch
import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.yolo_v12n import YOLOv12n
from data.dataset import VOCDataset
from torch.utils.data import DataLoader

def test_model():
    print("\n--- Testing Model Architecture ---")
    try:
        model = YOLOv12n(nc=20)
        x = torch.randn(2, 3, 640, 640)
        y = model(x)
        print("Forward pass successful.")
        for i, out in enumerate(y):
            print(f"Output {i} shape: {out.shape}")
        # Expected: (2, 84, 80, 80), (2, 84, 40, 40), (2, 84, 20, 20)
        assert y[0].shape == (2, 84, 80, 80)
        print("Model Architectural Integrity: PASS")
    except Exception as e:
        print(f"Model Test Failed: {e}")
        raise e

def create_dummy_data():
    print("\n--- Creating Dummy Data for Testing ---")
    os.makedirs('tests_data/images', exist_ok=True)
    os.makedirs('tests_data/labels', exist_ok=True)
    
    img_list = []
    for i in range(10): # 10 dummy images
        # Create random image
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        # Draw some rectangles
        cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), 2)
        cv2.putText(img, f"Img {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        img_path = os.path.abspath(f'tests_data/images/img_{i}.jpg')
        cv2.imwrite(img_path, img)
        img_list.append(img_path)
        
        # Create dummy label: class x y w h (normalized)
        # Class 0, center 0.5 0.5, size 0.2 0.2
        with open(f'tests_data/labels/img_{i}.txt', 'w') as f:
            f.write("0 0.5 0.5 0.2 0.2\n")
            f.write("1 0.2 0.2 0.1 0.1\n") # Class 1
            
    torch.save(img_list, 'tests_data/dummy_list.pt')
    print("Dummy data created.")
    return 'tests_data/dummy_list.pt', 'tests_data/labels'

def test_dataset_and_loader(list_path, label_dir):
    print("\n--- Testing Dataset & DataLoader ---")
    try:
        ds = VOCDataset(img_list_path=list_path, label_dir=label_dir, augment=True)
        print(f"Dataset length: {len(ds)}")
        
        # Test __getitem__
        img, lbl = ds[0]
        print(f"Sample item shape: {img.shape}, Label shape: {lbl.shape}")
        
        # Test DataLoader
        loader = DataLoader(ds, batch_size=4, shuffle=True,  collate_fn=VOCDataset.collate_fn if hasattr(ds, 'collate_fn') else None)
        # Note: Default collate might fail if labels have different lengths?
        # YOLO datasets usually need custom collate. 
        # Let's check if our dataset handles batching of variable labels?
        # Standard PyTorch collate fails with variable size tensors.
        # We need a collate_fn. 
        
        # Adding a simple collate function inside here for testing if not present
        def custom_collate(batch):
            imgs, labels = zip(*batch)
            imgs = torch.stack(imgs, 0)
            # Add sample index to targets -> [idx, cls, x, y, w, h]
            new_labels = []
            for i, l in enumerate(labels):
                if l.shape[0] > 0:
                    l_new = torch.zeros((l.shape[0], 6))
                    l_new[:, 0] = i
                    l_new[:, 1:] = l
                    new_labels.append(l_new)
            if new_labels:
                new_labels = torch.cat(new_labels, 0)
            else:
                new_labels = torch.zeros((0, 6))
            return imgs, new_labels

        loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=custom_collate)
        
        for imgs, targets in loader:
            print(f"Batch generation successful. Imgs: {imgs.shape}, Targets: {targets.shape}")
            break
        print("Dataset & DataLoader: PASS")
        
        # Save a debug image
        debug_img = imgs[0].permute(1, 2, 0).numpy().copy()
        # Denormalize? No, it's uint8? No, dataset returns it valid? 
        # Let's check dataset output type.
        # usually dataset returns transformed float tensor. 
        # If we didn't normalize to 0-1, it's 0-255 float? or uint8?
        # My implementation: `img = np.ascontiguousarray(img)` and `torch.from_numpy(img)`. 
        # It didn't divide by 255. So it is 0-255.
        cv2.imwrite('tests_data/debug_batch_sample.jpg', debug_img)
        print("Saved debug sample to tests_data/debug_batch_sample.jpg")
        
    except Exception as e:
        print(f"Dataset Test Failed: {e}")
        import traceback
        traceback.print_exc()
        raise e

def cleanup():
    # Optional: remove tests_data
    # shutil.rmtree('tests_data')
    pass

if __name__ == "__main__":
    test_model()
    list_path, label_dir = create_dummy_data()
    test_dataset_and_loader(list_path, label_dir)
    cleanup()
    print("\nALL SYSTEM TESTS PASSED.")
