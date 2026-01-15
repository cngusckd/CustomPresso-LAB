import torch
import numpy as np

def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    box1: (N, 4)
    box2: (M, 4)
    Returns: (N, M)
    """
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    
    return inter / (area1[:, None] + area2 - inter)

class MAPCalculator:
    def __init__(self, iou_vals=np.linspace(0.5, 0.95, 10)):
        self.iou_vals = iou_vals
        self.stats = [] # list of (correct, conf, cls)

    def update(self, preds, targets):
        """
        preds: list of [x1, y1, x2, y2, conf, cls] (N, 6)
        targets: list of [cls, x1, y1, x2, y2] (M, 5) - usually normalized?
        Let's assume targets are converted to pixel coords before update or we handle it here.
        Actually, trainer should pass pixel coords for both.
        preds: (N, 6) Tensor
        targets: (M, 5) Tensor, cls, x1, y1, x2, y2
        """
        if preds is None:
            if targets.shape[0] > 0:
                # All missed
                self.stats.append((torch.zeros(0, self.iou_vals.shape[0], dtype=torch.bool), 
                                   torch.empty(0), 
                                   torch.empty(0), 
                                   targets[:, 0].long()))
            return

        # Process batch
        # This update is usually called per image or batch. To keep it simple, let's assume per-image iteration in trainer 
        # or we handle batch here. 
        # Standard: Process one image at a time effectively.
        
        # If batch
        if isinstance(preds, list):
             # list of tensors?
             raise NotImplementedError("Pass single image preds/targets for now or handle loop outside")

        # Preds: [N, 6] (x1, y1, x2, y2, conf, cls)
        # Targets: [M, 5] (cls, x1, y1, x2, y2)
        
        device = preds.device
        
        if targets.shape[0] == 0:
            if preds.shape[0] > 0:
                # All false positives
                correct = torch.zeros(preds.shape[0], self.iou_vals.shape[0], dtype=torch.bool, device=device)
                self.stats.append((correct, preds[:, 4], preds[:, 5], torch.empty(0, device=device)))
            return

        iou = box_iou(targets[:, 1:], preds[:, :4]) # (M, N)
        
        correct = torch.zeros(preds.shape[0], self.iou_vals.shape[0], dtype=torch.bool, device=device)
        correct_class = targets[:, 0:1] == preds[:, 5] # (M, N) - broadcast? no
        # targets (M), preds(N)
        
        # Match predictions to targets
        # For each IOU threshold
        # We need to match highest IOU first
        
        for i, iouv in enumerate(self.iou_vals):
            x = torch.where((iou >= iouv) & (targets[:, 0:1] == preds[:, 5]))  # IoU > thresh AND Same Class
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                
                correct[matches[:, 1].astype(int), i] = True
        
        self.stats.append((correct.cpu(), preds[:, 4].cpu(), preds[:, 5].cpu(), targets[:, 0].long().cpu()))

    def compute(self):
        stats = [torch.cat(x, 0) for x in zip(*self.stats)]  # to tensor
        if len(stats) and stats[0].any():
            tp, conf, pred_cls, target_cls = stats
            
            # Per class AP
            unique_classes = torch.unique(target_cls)
            ap = []
            p, r = [], []
            
            for c in unique_classes:
                i = pred_cls == c
                n_l = (target_cls == c).sum().item() # number of labels
                n_p = i.sum().item() # number of predictions
                
                if n_p == 0 or n_l == 0:
                    ap.append(np.zeros(len(self.iou_vals)))
                    continue
                
                # Sort by confidence
                c_conf = conf[i]
                c_tp = tp[i]
                
                f_c = c_conf.argsort(descending=True)
                c_tp = c_tp[f_c]
                
                tp_cumsum = c_tp.cumsum(0).float()
                fp_cumsum = (1 - c_tp.float()).cumsum(0)
                
                # Recall
                recall = tp_cumsum / (n_l + 1e-16)
                # Precision
                precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
                
                # AP from P-R curve
                # Append sentinel values
                ap_c = []
                for j in range(len(self.iou_vals)):
                    # Compute AP@IoU[j]
                    p_ = precision[:, j]
                    r_ = recall[:, j]
                    
                    # 11-point interpolated AP or Continuous? 
                    # COCO uses 101-point? Or just AUC.
                    # Standard simple integration:
                    
                    # Add start and end
                    mrec = np.concatenate(([0.0], r_.numpy(), [1.0]))
                    mpre = np.concatenate(([1.0], p_.numpy(), [0.0]))
                    
                    # Compute the precision envelope
                    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
                    
                    # Integrate area under curve
                    method = 'resampled' # or 'continuous'
                    
                    # Exact integration
                    i_ = np.where(mrec[1:] != mrec[:-1])[0]
                    ap_val = np.sum((mrec[i_ + 1] - mrec[i_]) * mpre[i_ + 1])
                    ap_c.append(ap_val)
                    
                ap.append(ap_c)
            
            # AP is now (NC, 10)
            ap = np.array(ap)
            mAP50 = ap[:, 0].mean()
            mAP = ap.mean() # mean over classes and IoUs
            
            return mAP50, mAP
            
        return 0.0, 0.0
