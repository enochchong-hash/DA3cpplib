#!/usr/bin/env python3
"""Generate the tiny ONNX graph consumed by test_tensorrt.cpp."""

import argparse
import onnx
from onnx import TensorProto, helper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 14, 28])
    depth = helper.make_tensor_value_info("depth", TensorProto.FLOAT, [1, 14, 28])
    confidence = helper.make_tensor_value_info(
        "confidence", TensorProto.FLOAT, [1, 14, 28])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    one = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["image", "axes"], ["depth"], keepdims=0),
         helper.make_node("Add", ["depth", "one"], ["confidence"])],
        "da3_tensorrt_contract", [image], [depth, confidence], [axes, one])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
