#!/usr/bin/env python3
"""Export a fixed-resolution DA3 depth/confidence graph for TensorRT.

The exporter consumes an official Depth Anything 3 checkpoint directory and
source checkout. It deliberately exports only the single-view DualDPT depth
path used by da3::infer(..., with_pose=false); pose, rays, Gaussian
reconstruction, mono, nested, and DA2 remain on the GGUF backend.
"""

import argparse
import os
import sys
import types

import torch


def install_optional_dependency_stubs():
    # Plain model construction does not use video/GLB export or torchvision's
    # compiled operators, but importing the official API pulls them in.
    if "depth_anything_3.utils.export" not in sys.modules:
        module = types.ModuleType("depth_anything_3.utils.export")
        module.export = lambda *args, **kwargs: None
        sys.modules[module.__name__] = module
    if "torchvision" not in sys.modules:
        class IdentityTransform:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, value): return value

        class TransformModule(types.ModuleType):
            def __getattr__(self, _name): return IdentityTransform

        torchvision = types.ModuleType("torchvision")
        transforms = TransformModule("torchvision.transforms")
        torchvision.transforms = transforms
        sys.modules["torchvision"] = torchvision
        sys.modules["torchvision.transforms"] = transforms


class DepthGraph(torch.nn.Module):
    def __init__(self, network, height, width):
        super().__init__()
        self.backbone = network.backbone
        self.head = network.head
        self.height = height
        self.width = width

    def forward(self, image):
        # Official DA3 backbone input is [batch, views, channels, H, W].
        features, _ = self.backbone(
            image[:, None], cam_token=None, export_feat_layers=[],
            ref_view_strategy="saddle_balanced")
        output = self.head(list(features), self.height, self.width,
                           patch_start_idx=0)
        depth = output["depth"].reshape(image.shape[0], self.height, self.width)
        confidence = output["depth_conf"].reshape(
            image.shape[0], self.height, self.width)
        return depth, confidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="official DA3 checkpoint directory")
    parser.add_argument("output", help="output .onnx path")
    parser.add_argument("--source", help="official DA3 checkout (directory containing src/)")
    parser.add_argument("--height", type=int, default=504)
    parser.add_argument("--width", type=int, default=504)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    if args.height <= 0 or args.width <= 0 or args.height % 14 or args.width % 14:
        parser.error("height and width must be positive multiples of DA3's patch size (14)")
    if args.source:
        sys.path.insert(0, os.path.join(os.path.abspath(args.source), "src"))

    install_optional_dependency_stubs()
    from depth_anything_3.api import DepthAnything3

    wrapped = DepthAnything3.from_pretrained(args.model_dir)
    network = wrapped.model if hasattr(wrapped, "model") else wrapped
    graph = DepthGraph(network.eval().float(), args.height, args.width).eval()
    sample = torch.zeros(1, 3, args.height, args.width, dtype=torch.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            graph, sample, args.output, export_params=True, opset_version=args.opset,
            do_constant_folding=True, input_names=["image"],
            output_names=["depth", "confidence"], dynamic_axes=None,
            dynamo=False, external_data=True,
        )
    print(f"wrote {args.output}: image [1,3,{args.height},{args.width}] -> depth/confidence")


if __name__ == "__main__":
    main()
