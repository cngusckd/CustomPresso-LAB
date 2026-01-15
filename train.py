import torch
from models.yolo_v12n import YOLOv12n
from data.dataset import VOCDataset
from core.trainer import Trainer
from torch.utils.data import DataLoader
import argparse
import os

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16) # RTX 3070 8GB -> 16 maybe ok?
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data-path', type=str, default='./data/voc_images_list.pt')
    opt = parser.parse_args()

    # Model
    print("Initializing YOLOv12n...")
    model = YOLOv12n(nc=20)
    
    # Data
    print("Loading Data...")
    if not torch.cuda.is_available() and opt.device == 'cuda':
        print("Warning: CUDA not available, using CPU.")
        opt.device = 'cpu'
        
    try:
        train_path = './data/voc_train_list.pt'
        val_path = './data/voc_val_list.pt'
        
        # Check existence
        if not os.path.exists(train_path) or not os.path.exists(val_path):
             print(f"Error: Data files not found. Run 'python data/download_voc.py' first.")
             return

        print(f"Loading Training Data from {train_path}...")
        train_ds = VOCDataset(img_list_path=train_path, augment=True)
        
        print(f"Loading Validation Data from {val_path}...")
        val_ds = VOCDataset(img_list_path=val_path, augment=False)
        
        # Windows/WSL: num_workers=0 is safest to avoid IPC errors / shared memory issues
        train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, collate_fn=custom_collate, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=opt.batch_size, shuffle=False, collate_fn=custom_collate, num_workers=0)
        
        print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images")
        
        # Trainer
        trainer = Trainer(model, train_loader, val_loader, epochs=opt.epochs, lr=opt.lr, device=opt.device)
        trainer.train()
        
    except Exception as e:
        print(f"Error: {e}")
        # Create dummy data for dry run if real data fails
        print("Creating dummy dataset for verification...")
        pass

if __name__ == "__main__":
    main()
