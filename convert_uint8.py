"""
This keeps activations in FP32 but uses int8 for
weights, which works well on CPU.
"""

from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="funiegan_model_dynamic.onnx",
    model_output="funiegan_model_uint8.onnx",
    weight_type=QuantType.QUInt8
)

