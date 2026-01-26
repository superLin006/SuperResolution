#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export PyTorch EDSR model to ONNX format."""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import onnx
from onnxsim import simplify

# Import model definition from test_pytorch.py
from test_pytorch import EDSR


def export_onnx(pt_model_path, output_onnx_path, input_size=(256, 256), opset_version=11):
    """
    Export EDSR PyTorch model to ONNX.

    Args:
        pt_model_path: Path to .pt model file
        output_onnx_path: Path to save .onnx file
        input_size: Input image size (H, W), default 256x256
        opset_version: ONNX opset version
    """
    # Infer scale from model name
    scale = 4  # default
    if 'x2' in pt_model_path.lower():
        scale = 2
    elif 'x3' in pt_model_path.lower():
        scale = 3
    elif 'x4' in pt_model_path.lower():
        scale = 4

    print(f"Exporting EDSR model (scale={scale}) to ONNX...")
    print(f"Input size: {input_size}")

    # Build model
    model = EDSR(scale=scale)

    # Load state dict
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

    # Simplify ONNX model
    print("Simplifying ONNX model...")
    try:
        simplified_model, check = simplify(onnx_model)
        if check:
            onnx.save(simplified_model, output_onnx_path)
            print("ONNX model simplified successfully!")
        else:
            print("WARNING: ONNX simplification failed, using original model")
    except Exception as e:
        print(f"WARNING: ONNX simplification error: {e}")
        print("Using original ONNX model")

    # Check final model
    print("\nFinal ONNX model info:")
    onnx_model = onnx.load(output_onnx_path)
    print(f"  Input: {onnx_model.graph.input[0].name}")
    print(f"  Output: {onnx_model.graph.output[0].name}")
    print(f"  Graph nodes: {len(onnx_model.graph.node)}")

    print(f"\nONNX model saved to: {output_onnx_path}")
    return output_onnx_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_onnx.py <pt_model_path> [output_onnx_path] [input_height] [input_width]")
        print("Example: python export_onnx.py ../model/EDSR_x4.pt ../model/edsr_x4.onnx 256 256")
        sys.exit(1)

    pt_model_path = sys.argv[1]

    # Default output path
    if len(sys.argv) > 2:
        output_onnx_path = sys.argv[2]
    else:
        base_name = os.path.splitext(os.path.basename(pt_model_path))[0]
        output_onnx_path = os.path.join(os.path.dirname(pt_model_path), f"{base_name}.onnx")

    # Input size
    input_h = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    input_w = int(sys.argv[4]) if len(sys.argv) > 4 else 256

    export_onnx(pt_model_path, output_onnx_path, (input_h, input_w))


if __name__ == "__main__":
    main()
