import onnx
from float16 import convert_float_to_float16_model_path

# Load and convert model
model_fp16 = convert_float_to_float16_model_path(
    "funiegan_model_dynamic.onnx",    # your input model path
    keep_io_types=True                # keep input/output float32
)

# Save the FP16 model
onnx.save(model_fp16, "funiegan_model_fp16.onnx")
