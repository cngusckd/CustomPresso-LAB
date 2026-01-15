import torch
import torch.nn as nn
from models.yolo_v12n import Conv

def fuse_conv_bn(model):
    """
    Fuses Conv2d and BatchNorm2d layers in the model for faster inference.
    Iterates through the model modules and fuses valid Conv->BN pairs.
    """
    print("Fusing Conv2d and BatchNorm2d layers...")
    for m in model.modules():
        if isinstance(m, Conv) and hasattr(m, 'bn') and hasattr(m, 'conv'):
            # m is our custom Conv wrapper which has .conv and .bn
            # Fuse
            conv = m.conv
            bn = m.bn
            
            fused_conv = torch.nn.utils.fuse_conv_bn_eval(conv, bn)
            
            # Update the Conv wrapper
            m.conv = fused_conv
            m.bn = nn.Identity() # Remove BN
            
            # If the wrapper has forward logic using .bn, it ensures Identity is pass-through.
            # our Conv.forward is: act(bn(conv(x)))
            # With fused: act(Identity(fused_conv(x))) -> act(fused_conv(x)) - Correct.
            
    # Also look for standard nn.Sequential patterns if any (e.g. in Head)
    # The models/yolo_v12n.py uses custom Conv mainly.
    # But Detect head uses nn.Sequential(Conv, Conv, nn.Conv2d).
    # Those inner Convs already handled above.
    
    return model
