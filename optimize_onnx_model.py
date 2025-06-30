from onnxruntime.transformers.optimizer import optimize_model

optimized_model = optimize_model(
    "funiegan_model_dynamic.onnx",
    model_type="bert",  # Not important for generic models
    num_heads=0,        # Ignore for non-transformers
    hidden_size=0
)

optimized_model.save_model_to_file("funiegan_model_optimized.onnx")

print("Optimized model saved.")
