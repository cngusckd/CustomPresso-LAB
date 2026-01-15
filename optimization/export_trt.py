import torch
import torch.onnx
import os
import sys

def export_onnx(model, dummy_input, onnx_path="model.onnx", opset=13):
    """
    Export PyTorch model to ONNX.
    """
    print(f"Exporting to ONNX: {onnx_path}")
    model.eval()
    
    # Dynamic axes
    dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        verbose=False, 
        opset_version=opset,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    print("ONNX export successful.")
    return onnx_path

def build_trt_engine(onnx_path, engine_path, fp16=True, int8=False):
    """
    Build TensorRT engine from ONNX.
    Requires tensorrt python package.
    """
    print(f"Building TensorRT Engine: {engine_path} (FP16={fp16}, INT8={int8})")
    
    try:
        import tensorrt as trt
    except ImportError:
        print("Error: 'tensorrt' package not found. Skipping engine build.")
        print("To install: pip install tensorrt")
        return None

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # Flags logic for different TRT versions
    # Modern TRT: create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    config = builder.create_builder_config()
    
    # Memory pool limit (workspace) - deprecated in newer TRT, passing directly?
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
    config.max_workspace_size = 1 << 30 # For older TRT
    
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # INT8 requires calibration. Using dummy logic or calibration cache?
        # Without calibration cache, it assumes QAT ONNX or ranges present?
        # PyTorch QAT export adds QuantizeLinear/DequantizeLinear nodes which TRT understands.
        # If PTQ via TRT, we need Calibrator class.
        print("Warning: INT8 build requested but no Calibrator provided. Assuming QAT ONNX or falling back.")
    
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            
    # Build
    # Runtime
    try:
        plan = builder.build_serialized_network(network, config)
        if plan is None:
             print("Engine build failed!")
             return None
             
        with open(engine_path, "wb") as f:
            f.write(plan)
            
        print("Engine built successfully.")
        return engine_path
        
    except AttributeError:
        # Older TRT API fallback (build_engine)
        engine = builder.build_engine(network, config)
        if engine:
            with open(engine_path, "wb") as f:
                f.write(engine.serialize())
            return engine_path
            
    return None

if __name__ == "__main__":
    # Dummy usage
    from models.yolo_v12n import YOLOv12n
    m = YOLOv12n()
    x = torch.randn(1, 3, 640, 640)
    export_onnx(m, x, "test.onnx")
