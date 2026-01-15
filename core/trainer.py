import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import os
import copy
from .loss import ComputeLoss
from .metrics import MAPCalculator

class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    """
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        
        msd = model.state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point: 
                v *= d
                v += (1 - d) * msd[k].detach()

class Trainer:
    def __init__(self, model, train_loader, val_loader=None, epochs=100, lr=0.01, device='cuda', save_dir='./runs/train'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=5e-4)
        
        # Scheduler
        # OneCycleLR
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
        
        # Loss
        self.compute_loss = ComputeLoss(self.model)
        
        # EMA
        self.ema = ModelEMA(self.model)
        
        # Scalar
        self.use_cuda = device.startswith('cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_cuda)
        
        self.best_map = 0.0

    def train(self):
        print(f"Starting training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", ncols = 150)
            
            epoch_loss = 0
            for i, (imgs, targets) in enumerate(pbar):
                imgs = imgs.to(self.device) # Already scaled 0-1 in dataset
                targets = targets.to(self.device)
                
                # Forward
                # Device specific autocast
                if self.use_cuda:
                     with torch.amp.autocast('cuda', enabled=True):
                        pred = self.model(imgs)
                        loss = self.compute_loss(pred, targets)
                else:
                     pred = self.model(imgs)
                     loss = self.compute_loss(pred, targets)
                
                # Backward
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Clip gradients
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # EMA
                self.ema.update(self.model)
                
                # Scheduler
                self.scheduler.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
            
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
            
            # Validation
            if self.val_loader:
                map50, map = self.validate()
                print(f"Val mAP@50: {map50:.3f}, mAP@50:95: {map:.3f}")
                
                if map50 > self.best_map:
                    self.best_map = map50
                    self.save_checkpoint('best.pt')
            
            self.save_checkpoint('last.pt')

    def validate(self):
        self.ema.ema.eval()
        pbar = tqdm(self.val_loader, desc="Validating")
        metric = MAPCalculator()
        
        with torch.no_grad():
            for imgs, targets in pbar:
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                
                # Preds
                preds = self.ema.ema(imgs) # list of [N, 6] tensors
                
                # Update metrics
                # Targets: [M, 6] (img_idx, cls, x, y, w, h) normalized?
                # MAPCalculator expects targets as [M, 5] (cls, x1, y1, x2, y2) pixels?
                # My Dataset returns targets as [img_idx, cls, x, y, w, h] normalized (if using collate_fn I wrote in test)
                # Wait, test_framework collate adds img_idx -> [idx, cls, x, y, w, h].
                # Dataset __getitem__ returns [N, 5].
                # Trainer loop gets batch.
                
                for i in range(len(imgs)):
                    # Get targets for this image
                    t = targets[targets[:, 0] == i]
                    if len(t) > 0:
                        # Convert xywh norm -> xyxy pixels
                        # t: [num_obj, 6] -> slice [num_obj, 2:6]
                        cls_labels = t[:, 1]
                        bboxes = t[:, 2:]
                        
                        # Scale
                        h, w = imgs.shape[2:]
                        bboxes[:, 0] *= w
                        bboxes[:, 2] *= w
                        bboxes[:, 1] *= h
                        bboxes[:, 3] *= h
                        
                        # xywh -> xyxy
                        x1 = bboxes[:, 0] - bboxes[:, 2] / 2
                        y1 = bboxes[:, 1] - bboxes[:, 3] / 2
                        x2 = bboxes[:, 0] + bboxes[:, 2] / 2
                        y2 = bboxes[:, 1] + bboxes[:, 3] / 2
                        
                        target_formatted = torch.stack([cls_labels, x1, y1, x2, y2], 1)
                    else:
                        target_formatted = torch.empty(0, 5, device=self.device)
                    
                    metric.update(preds[i], target_formatted)
        
        return metric.compute()

    def save_checkpoint(self, filename):
        ckpt = {
            'model': self.model.state_dict(),
            'ema': self.ema.ema.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(ckpt, os.path.join(self.save_dir, filename))
