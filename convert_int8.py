"""
This keeps activations in FP32 but uses int8 for
weights, which works well on CPU.
"""
# import onnx
# from onnxruntime.quantization import quantize_dynamic, QuantType
#
# quantize_dynamic(
#     model_input="funiegan_model_dynamic.onnx",
#     model_output="funiegan_model_int8.onnx",
#     weight_type=QuantType.QInt8,
#     per_channel=True,
#     reduce_range=True
# )

# Check model (check how much was quantized)
# model = onnx.load("funiegan_model_int8.onnx")
# for tensor in model.graph.initializer:
#     print(tensor.name, onnx.TensorProto.DataType.Name(tensor.data_type))
