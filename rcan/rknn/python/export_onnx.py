#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export PyTorch RCAN model to ONNX format."""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import onnx
from onnxsim import simplify

# Import model definition from test_pytorch.py
from test_pytorch import RCAN


def infer_scale_from_model_name(model_path):
    """Infer scale factor from model filename."""
    lower = model_path.lower()
    for s in (2, 3, 4, 8):
        if f"x{s}" in lower or f"bix{s}" in lower or f"bdx{s}" in lower:
            return s
    return 4  # default


def export_onnx(pt_model_path, output_onnx_path, input_size=(256, 256), opset_version=11):
    """
    Export RCAN PyTorch model to ONNX.

    Args:
        pt_model_path: Path to .pt model file
        output_onnx_path: Path to save .onnx file
        input_size: Input image size (H, W), default 256x256
        opset_version: ONNX opset version

    Note:
        RCAN uses PixelShuffle for upsampling which converts to DepthToSpace in ONNX.
        RKNN supports DepthToSpace, so this should work directly.
    """
    # Infer scale from model name
    scale = infer_scale_from_model_name(pt_model_path)

    print(f"Exporting RCAN model (scale={scale}) to ONNX...")
    print(f"Input size: {input_size}")

    # Build model
    model = RCAN(scale=scale)

    # Load state dict
    print(f"Loading PyTorch model from {pt_model_path}...")
    state = torch.load(pt_model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    # Remove 'module.' prefix if exists
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=True)
    model.eval()

    # Create dummy input
    h, w = input_size
    dummy_input = torch.randn(1, 3, h, w)

    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    expected_h = h * scale
    expected_w = w * scale
    if output.shape == (1, 3, expected_h, expected_w):
        print(f"✓ Output shape matches expected scale x{scale}")
    else:
        print(f"⚠ Warning: Expected (1, 3, {expected_h}, {expected_w}), got {output.shape}")

    # Export to ONNX
    # Note: RKNN doesn't support dynamic shapes, so we export with fixed shape
    print(f"Exporting to {output_onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
        # No dynamic_axes for RKNN compatibility
    )
    print("ONNX export completed!")

    # Verify ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

    # Check model structure
    print("\nONNX model structure:")
    print(f"  Input: {onnx_model.graph.input[0].name}")
    input_shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    print(f"  Input shape: {input_shape}")
    print(f"  Output: {onnx_model.graph.output[0].name}")
    output_shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]
    print(f"  Output shape: {output_shape}")
    print(f"  Graph nodes: {len(onnx_model.graph.node)}")

    # Check for operators that might not be supported by RKNN
    print("\nChecking operators in ONNX graph...")
    op_set = set()
    for node in onnx_model.graph.node:
        op_set.add(node.op_type)

    # Read supported operators from RKNN documentation
    print(f"Total unique operators: {len(op_set)}")
    print("Operators used:", ", ".join(sorted(op_set)))

    # Known potentially unsupported operators
    unsupported_ops = []
    for op in op_set:
        if op in ["GlobalAveragePool", "ReduceMean"]:
            # These are supported but with restrictions
            print(f"  ⚠ {op}: May have batch_size restrictions in RKNN")
        elif op not in ["Conv", "Add", "Mul", "Relu", "Clip", "DepthToSpace", "Concat",
                       "Reshape", "Transpose", "Sigmoid", "Flatten", "Constant"]:
            print(f"  ? {op}: Please verify RKNN support")

    # Simplify ONNX model
    print("\nSimplifying ONNX model...")
    try:
        simplified_model, check = simplify(onnx_model)
        if check:
            onnx.save(simplified_model, output_onnx_path)
            print("✓ ONNX model simplified successfully!")

            # Re-check after simplification
            onnx_model = onnx.load(output_onnx_path)
            op_set = set(node.op_type for node in onnx_model.graph.node)
            print(f"Operators after simplification: {len(op_set)}")
        else:
            print("⚠ Warning: ONNX simplification check failed, using original model")
    except Exception as e:
        print(f"⚠ Warning: ONNX simplification error: {e}")
        print("Using original ONNX model")

    # Final model info
    print("\n" + "=" * 60)
    print(f"ONNX export completed successfully!")
    print(f"Model saved to: {output_onnx_path}")
    print(f"File size: {os.path.getsize(output_onnx_path) / (1024*1024):.2f} MB")
    print("=" * 60)

    return output_onnx_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_onnx.py <pt_model_path> [output_onnx_path] [input_height] [input_width]")
        print("Example: python export_onnx.py ../model/RCAN_BIX4.pt ../model/rcan_x4.onnx 256 256")
        print("\nNote: RKNN requires fixed input size. Default is 256x256.")
        sys.exit(1)

    pt_model_path = sys.argv[1]

    # Check if model exists
    if not os.path.exists(pt_model_path):
        print(f"Error: Model file not found: {pt_model_path}")
        sys.exit(1)

    # Default output path
    if len(sys.argv) > 2:
        output_onnx_path = sys.argv[2]
    else:
        base_name = os.path.splitext(os.path.basename(pt_model_path))[0]
        base_name = base_name.replace("BIX", "x").replace("BDX", "x")  # Normalize naming
        output_onnx_path = os.path.join(os.path.dirname(pt_model_path), f"{base_name}.onnx")

    # Input size (default 256x256 for RK3576)
    input_h = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    input_w = int(sys.argv[4]) if len(sys.argv) > 4 else 256

    export_onnx(pt_model_path, output_onnx_path, (input_h, input_w))


if __name__ == "__main__":
    main()
