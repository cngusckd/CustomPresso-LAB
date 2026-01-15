import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, groups, kernels, and expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class AreaAttention(nn.Module):
    """
    Area Attention module (simplified implementation).
    Uses scaled dot-product attention (FlashAttention where available).
    """
    def __init__(self, c1, c2, head_dim=64, p=0.0):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = c2 // head_dim
        self.scale = head_dim ** -0.5
        self.qkv = Conv(c1, c2 * 3, 1, 1)
        self.proj = Conv(c2, c2, 1, 1)
        self.p = p

    def forward(self, x):
        B, C, H, W = x.shape
        # Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W).permute(1, 0, 2, 4, 3) # 3, B, H, N, D
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # Dropot if needed
        
        x_attn = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
        return self.proj(x_attn)

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.
        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4GF?) Defaulting to 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        
        return self._predict(x)

    def _predict(self, x):
        # x is list of 3 tensors [B, C, H, W]
        # Decode and post-process
        # Check if anchors are initialized
        if self.anchors.shape[0] == 0 or self.anchors.device != x[0].device:
             self._make_anchors(x)
        
        # Concat all heads
        # x[i]: [B, 4*reg+nc, H, W]
        y = []
        for i, val in enumerate(x):
            b, c, h, w = val.shape
            # val = [B, 4+nc, H*W]
            val = val.view(b, c, -1).permute(0, 2, 1)
            y.append(val)
        
        y = torch.cat(y, 1) # [B, TotalAnchors, C]
        
        # Split box and cls
        box, cls = y.split((self.reg_max * 4, self.nc), 2)
        
        # DFL to distance
        # box: [B, A, 4, 16]
        b, a, _ = box.shape
        box = box.view(b, a, 4, self.reg_max).softmax(3)
        # We need self.dfl.conv to be accessible or use simple matmul if dfl is identity?
        # self.dfl is Module. 
        # But we need projection tensor. 
        # Create it dynamically or use self.dfl.conv.weight logic
        # DFL layer has conv(c1, 1). Weight is parameter [1, c1, 1, 1] 0..15
        
        # dfl(box) -> [B, 4, A] -> transpose to [B, A, 4]
        # But box is [B, A, 4, 16].
        # DFL consumes [B, C, A]. 
        # Let's use matmul logic like in loss.py as it is cleaner.
        if self.dfl.conv.weight.device != box.device:
             self.dfl.to(box.device)
             
        proj = torch.arange(self.reg_max, dtype=torch.float, device=box.device)
        dist = box.matmul(proj) # [B, A, 4] ltrb
        
        # Decode to xyxy
        # anchor_points: [A, 2]
        # dist: l, t, r, b
        # x1 = cx - l, y1 = cy - t, x2 = cx + r, y2 = cy + b
        
        lt, rb = dist.chunk(2, 2)
        x1y1 = self.anchors - lt 
        x2y2 = self.anchors + rb
        # Note: self.anchors are in stride units? Or pixel?
        # _make_anchors usually produces anchors in stride coordinates if we multiply later?
        # Or pixel?
        # Standard: anchors in grid coords (stride units) if we multiply by stride later.
        
        boxes = torch.cat((x1y1, x2y2), 2) # [B, A, 4]
        boxes = boxes * self.strides # Scale to pixels
        
        # Cls
        cls = cls.sigmoid()
        
        # Concat [B, N, 6] (x1, y1, x2, y2, conf, cls_id) ???
        # NMS usually expects [x1, y1, x2, y2, score]
        # We return [x1, y1, x2, y2, score, cls_ind]
        
        # We need NMS here? Or return raw?
        # Validation usually requires NMS.
        # I'll implement basic batched NMS.
        
        return self._nms(boxes, cls)

    def _make_anchors(self, feats):
        anchor_points, stride_tensor = [], []
        dtype, device = feats[0].dtype, feats[0].device
        for i, stride in enumerate([8, 16, 32]): # Hardcoded strides for now
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + 0.5
            sy = torch.arange(end=h, device=device, dtype=dtype) + 0.5
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        
        self.anchors = torch.cat(anchor_points)
        self.strides = torch.cat(stride_tensor)

    def _nms(self, bboxes, scores, conf_thres=0.001, iou_thres=0.6):
        # bboxes: [B, A, 4]
        # scores: [B, A, NC]
        
        # Filter low conf
        # Max score per anchor
        max_scores, labels = scores.max(2) # [B, A]
        
        output = []
        for b in range(bboxes.shape[0]):
            mask = max_scores[b] > conf_thres
            if not mask.any():
                output.append(torch.zeros(0, 6, device=bboxes.device))
                continue
            
            box = bboxes[b][mask]
            score = max_scores[b][mask]
            lbl = labels[b][mask]
            
            # NMS
            # torchvision.ops.nms
            import torchvision
            keep = torchvision.ops.nms(box, score, iou_thres)
            
            # Limit detections
            if keep.shape[0] > 300:
                keep = keep[:300]
                
            box = box[keep]
            score = score[keep]
            lbl = lbl[keep]
            
            # [N, 6] -> x1, y1, x2, y2, score, cls
            out = torch.cat((box, score[:, None], lbl[:, None].float()), 1)
            output.append(out)
        
        return output


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize DFL module with input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies DFL to input tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class YOLOv12n(nn.Module):
    def __init__(self, nc=20): # PASCAL VOC has 20 classes
        super().__init__()
        self.nc = nc
        
        # Backbone (YOLOv8-like + AreaAttention)
        # P1: 640 -> 320
        self.conv1 = Conv(3, 16, 3, 2) 
        # P2: 320 -> 160
        self.conv2 = Conv(16, 32, 3, 2)
        self.c2f1 = C2f(32, 32, 1, True)
        # P3: 160 -> 80
        self.conv3 = Conv(32, 64, 3, 2)
        self.c2f2 = C2f(64, 64, 2, True)
        # P4: 80 -> 40
        self.conv4 = Conv(64, 128, 3, 2)
        self.c2f3 = C2f(128, 128, 2, True)
        # P5: 40 -> 20
        self.conv5 = Conv(128, 256, 3, 2)
        self.c2f4 = C2f(256, 256, 1, True)
        self.sppf = SPPF(256, 256, 5)

        # Head (FPN + PAN)
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # C2f for Neck
        self.c2f_neck1 = C2f(384, 128, 1) # 256 + 128
        self.c2f_neck2 = C2f(192, 64, 1) # 128 + 64

        self.conv_down1 = Conv(64, 64, 3, 2)
        self.c2f_neck3 = C2f(192, 128, 1) # 64 + 128
        self.conv_down2 = Conv(128, 128, 3, 2)
        self.c2f_neck4 = C2f(384, 256, 1) # 128 + 256

        # Detect Head
        self.detect = Detect(nc=nc, ch=[64, 128, 256]) # P3, P4, P5

    def forward(self, x):
        # Backbone
        p1 = self.conv1(x)
        p2 = self.c2f1(self.conv2(p1))
        p3 = self.c2f2(self.conv3(p2))
        p4 = self.c2f3(self.conv4(p3))
        p5 = self.sppf(self.c2f4(self.conv5(p4)))

        # Neck (Top-Down)
        up_p5 = self.up_sample(p5)
        cat_p4 = torch.cat([up_p5, p4], dim=1) # 256 + 128 = 384
        head_p4 = self.c2f_neck1(cat_p4) # -> 128

        up_p4 = self.up_sample(head_p4)
        cat_p3 = torch.cat([up_p4, p3], dim=1) # 128 + 64 = 192
        head_p3 = self.c2f_neck2(cat_p3) # -> 64

        # Neck (Bottom-Up)
        down_p3 = self.conv_down1(head_p3)
        cat_down_p3 = torch.cat([down_p3, head_p4], dim=1) # 64 + 128 = 192
        out_p4 = self.c2f_neck3(cat_down_p3) # -> 128

        down_p4 = self.conv_down2(out_p4)
        cat_down_p4 = torch.cat([down_p4, p5], dim=1) # 128 + 256 = 384
        out_p5 = self.c2f_neck4(cat_down_p4) # -> 256

        return self.detect([head_p3, out_p4, out_p5])

if __name__ == "__main__":
    model = YOLOv12n(nc=20)
    print("Model created successfully.")
    x = torch.randn(1, 3, 640, 640)
    y = model(x)
    for i in y:
        print(i.shape)
