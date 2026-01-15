import torch
import torch.ao.quantization as quant
from torch.utils.data import DataLoader
from .graph_utils import fuse_conv_bn
import os

def quantize_ptq(model, calibration_loader, save_path=None):
    """
    Perform Post-Training Static Quantization (Int8).
    """
    print("Starting Post-Training Quantization (PTQ)...")
    model.eval()
    
    # 1. Fuse (Disabled for robustness)
    # model = fuse_conv_bn(model)
    
    # 2. Configure
    backend = 'x86' if next(model.parameters()).device.type == 'cpu' else 'fbgemm'
    model.qconfig = None
    
    # Strict selective: Only Conv2d and standard activations
    qconfig = quant.get_default_qconfig(backend)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
             # Check if it has weight (Depthwise might be issue? No, standard)
             module.qconfig = qconfig
             
    print(f"Using qconfig: {backend} (Conv2d Only)")
    
    # 3. Prepare
    print("Preparing model for quantization...")
    quant.prepare(model, inplace=True)
    
    # 4. Calibrate
    print("Calibrating with data...")
    device = next(model.parameters()).device
    # Ensure CPU for calibration if backend is x86
    model.to('cpu')
    
    with torch.no_grad():
        for i, (imgs, _) in enumerate(calibration_loader):
            if i > 5: break # Calibration with small subset
            imgs = imgs.to('cpu').float() / 255.0
            model(imgs)
            
    # 5. Convert
    print("Converting to Int8...")
    model.to('cpu')
    quant.convert(model, inplace=True)
    
    print("Quantization complete.")
    
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Quantized model saved to {save_path}")
        
    return model
