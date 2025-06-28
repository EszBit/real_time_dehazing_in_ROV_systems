"""
script used for converting .pth model to .onnx
"""

import torch
from nets.funiegan import GeneratorFunieGAN

model = GeneratorFunieGAN()
model.load_state_dict(torch.load("models/finetuned_color/generator_final.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 3, 256, 256)
#torch.onnx.export(model, dummy_input, "funiegan.onnx", input_names=['input'], output_names=['output'], opset_version=11)

torch.onnx.export(
    model,
    dummy_input,
    "funiegan_model_dynamic.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
    opset_version=11
)

