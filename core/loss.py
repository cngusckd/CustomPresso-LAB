import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    return iou

class ComputeLoss:
    def __init__(self, model, autobalance=False):
        super().__init__()
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp if hasattr(model, 'hyp') else {'box': 7.5, 'cls': 0.5, 'dfl': 1.5} # hyperparameters

        # Define criteria
        self.BCEcls = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = model.detect.stride if hasattr(model.detect, 'stride') else torch.tensor([8., 16., 32.])
        # Stride needs to be on device?
        # model.detect might not be accessible if wrapped?
        # We assume model is passed.
        self.nc = model.nc
        self.no = model.detect.no
        self.reg_max = model.detect.reg_max
        self.device = device
        
        # Project DFL
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = out[..., 1:5].mul_(scale_tensor)
        return out

    def __call__(self, preds, targets, imgs=None):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds
        batch_size = feats[0].shape[0]
        
        # 1. Generate Anchors
        dtype = feats[0].dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device) * self.stride[0]
        anchor_points, stride_tensor = [], []
        for i, stride in enumerate(self.stride):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=self.device, dtype=dtype) + 0.5
            sy = torch.arange(end=h, device=self.device, dtype=dtype) + 0.5
            sy, sx = torch.meshgrid(sy, sx) 
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=self.device))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        
        # 2. Decode Predictions
        pred_distri, pred_scores = [], []
        for i, pred in enumerate(feats):
            b, c, h, w = pred.shape
            p = pred.view(b, c, -1).permute(0, 2, 1)
            pred_distri.append(p[..., :self.reg_max * 4])
            pred_scores.append(p[..., self.reg_max * 4:])
        pred_distri = torch.cat(pred_distri, 1)
        pred_scores = torch.cat(pred_scores, 1)

        # Decode Bbox
        pred_distri_sf = pred_distri.view(batch_size, -1, 4, self.reg_max).softmax(3)
        pred_distri_val = pred_distri_sf.matmul(self.proj)
        x1 = anchor_points[:, 0] - pred_distri_val[..., 0]
        y1 = anchor_points[:, 1] - pred_distri_val[..., 1]
        x2 = anchor_points[:, 0] + pred_distri_val[..., 2]
        y2 = anchor_points[:, 1] + pred_distri_val[..., 3]
        pred_bboxes = torch.stack([x1, y1, x2, y2], -1) * stride_tensor

        # 3. Assignment & Loss
        target_cls_sum = 0
        
        for b in range(batch_size):
            t_idx = targets[:, 0] == b
            if not t_idx.any():
                continue
            
            t_box = targets[t_idx, 2:] # xywh normalized
            t_cls = targets[t_idx, 1].long()
            
            # De-normalize
            # targets are xywh. Convert to xyxy pixel
            t_box_pixel = t_box.clone()
            t_box_pixel[:, 0] *= imgsz[1] # x
            t_box_pixel[:, 2] *= imgsz[1]
            t_box_pixel[:, 1] *= imgsz[0] # y
            t_box_pixel[:, 3] *= imgsz[0]
            t_box_xyxy = xywh2xyxy(t_box_pixel)

            # Assign: Center Sampling
            # Check if anchor center is close to GT center
            t_centers = t_box_pixel[:, :2] # xy
            
            # Dist matrix: [A, N]
            # Expanding to avoid huge memory? No, typically A ~ 8400, N ~ 10-50. Small.
            # Only consider anchors within 2.5*stride of center
            
            # Simple Assignment (TopK) is safer.
            # But let's stick to a robust simple one: 
            # Find anchors inside GT boxes.
            
            # A xy
            anchors_xy = anchor_points * stride_tensor
            
            # Matrix [A, N, 4]
            # gt_x1, gt_y1, gt_x2, gt_y2
            lt = anchors_xy[:, None, :] - t_box_xyxy[None, :, :2] # xy - x1y1
            rb = t_box_xyxy[None, :, 2:] - anchors_xy[:, None, :] # x2y2 - xy
            
            inside_mask = (torch.min(lt, rb).min(2)[0] > 0) # [A, N]
            
            # If anchor inside multiple, pick max IoU?
            # Pick one with max IoU
            # Need Pred IoU with GT
            
            if inside_mask.sum() == 0:
                continue

            # Compute IoU of preds with GTs
            # pred_bboxes[b]: [A, 4]
            # t_box_xyxy: [N, 4]
            
            # We only compute logic for 'inside' anchors to save compute?
            # Full implementation usually computes Alignment Metric = s^alpha * u^beta
            # Let's simple: Just matches
            
            mask_idx = inside_mask.nonzero(as_tuple=True) # (anchor_idx, gt_idx)
            if mask_idx[0].shape[0] == 0:
                 continue
                 
            # Resolve ambiguity: Anchor assigned to only 1 GT (max IoU)
            # Simplification: Just take the first valid match or basic
            # Let's take the one with better IoU
            
            # Implementation of basic loss on matched anchors
            a_idx = mask_idx[0]
            g_idx = mask_idx[1]
            
            # Calculate IoU for these pairs
            p_b = pred_bboxes[b, a_idx] # [K, 4]
            g_b = t_box_xyxy[g_idx]     # [K, 4]
            ious = bbox_iou(p_b.T, g_b.T, x1y1x2y2=True, CIoU=True) # [K]
            
            # Box Loss
            loss[0] += (1.0 - ious).sum()
            
            # Class Loss
            # Target scores: 1.0 (or IoU?) 
            # YOLOv8 uses IoU as soft target? no, usually 1.0 logic unless VFL.
            # BCE.
            t_scores = torch.zeros(a_idx.shape[0], self.nc, device=self.device)
            t_scores[torch.arange(a_idx.shape[0]), t_cls[g_idx]] = 1.0 # Hard target
            
            p_scores = pred_scores[b, a_idx] # [K, nc] (logits)
            loss[1] += self.BCEcls(p_scores, t_scores).sum()
            
            target_cls_sum += a_idx.shape[0]
            
            # DFL Loss
            # anchors: anchor_points[a_idx] (cx, cy)
            # stride: stride_tensor[a_idx]
            # Target dist: (gt - anchor) / stride
            # t_box_xyxy[g_idx] is GT
            
            # l = (anchor.x - gt.x1) / stride
            # t = (anchor.y - gt.y1) / stride
            # r = (gt.x2 - anchor.x) / stride
            # b = (gt.y2 - anchor.y) / stride
            
            anc = anchor_points[a_idx]
            st = stride_tensor[a_idx]
            
            # dist target
            target_ltrb = torch.stack([
                anc[:, 0] - g_b[:, 0] / st[:, 0],
                anc[:, 1] - g_b[:, 1] / st[:, 0],
                g_b[:, 2] / st[:, 0] - anc[:, 0],
                g_b[:, 3] / st[:, 0] - anc[:, 1],
            ], -1).clamp(0, self.reg_max - 0.01)
            
            # DFL calculation (Distribution Focal Loss)
            # We have pred_distri[b, a_idx] [K, 4*reg_max]
            # We need to reshape to [K, 4, reg_max]
            # Standard DFL loss function...
            
            # Simplified DFL:
            # target_ltrb is float. We need label left and right.
            tl = target_ltrb.long()
            tr = tl + 1
            wl = tr.float() - target_ltrb
            wr = target_ltrb - tl.float()
            
            dfl_view = pred_distri[b, a_idx].view(-1, 4, self.reg_max)
            # Cross Entropy against tl and tr
            # loss = F.cross_entropy(...) * wl + ... 
            
            loss[2] += (F.cross_entropy(dfl_view, tl, reduction='none').view(-1, 4) * wl[:,:,None].squeeze() + \
                       F.cross_entropy(dfl_view, tr, reduction='none').view(-1, 4) * wr[:,:,None].squeeze()).mean(-1).sum()

        if target_cls_sum > 0:
            loss /= target_cls_sum
        
        # Ensure gradient attachment
        loss_sum = loss.sum()
        if not loss_sum.requires_grad:
             # Attach to predictions keys
             loss_sum = loss_sum + (pred_scores.sum() * 0) + (pred_distri.sum() * 0)
        
        return loss_sum

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y
