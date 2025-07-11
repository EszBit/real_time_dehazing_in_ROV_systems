"""
FastAPI wrapper for FUnIE-GAN ONNX model (dynamic input size).

This server exposes a single `/predict/` endpoint that accepts an image upload (JPEG, PNG, etc.)
and optional target resolution (`width`, `height`). It loads the model using ONNX Runtime with
CoreML or CPU execution, applies necessary preprocessing and padding, runs inference, and returns
an enhanced image (PNG) as response.

Use case: deployed as a lightweight image enhancement backend, callable from a Rust-based interface.
"""
from fastapi import FastAPI, File, UploadFile, Query
import uvicorn
import onnxruntime as ort
from PIL import Image
import numpy as np
import io
from fastapi.responses import Response

app = FastAPI()

# Load ONNX model with CoreML + CPU fallback
sess_options = ort.SessionOptions()
session = ort.InferenceSession(
    "funiegan_model_dynamic.onnx",
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
    sess_options=sess_options
)
input_name = session.get_inputs()[0].name

# Debug print: list active providers
print("Available providers:", ort.get_available_providers())
print("Active provider(s):", session.get_providers())

# Endpoint to receive image + optional resize
@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    width: int = Query(None, description="Resize width"),
    height: int = Query(None, description="Resize height")
):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Resize if width and height are given
    if width is not None and height is not None:
        try:
            img = img.resize((int(width), int(height)), resample=Image.BICUBIC)
        except Exception as e:
            return {"error": f"Failed to resize image: {str(e)}"}

    # Pad image to multiple of 32 (required for model)
    img = pad_to_multiple(img, 32)

    # Preprocess image: normalize and reorder dims
    img_np = (np.array(img).astype(np.float32) / 255.0 - 0.5) / 0.5
    img_np = np.transpose(img_np, (2, 0, 1))[None, :, :, :]  # (1, 3, H, W)

    # Run model inference
    output = session.run(None, {input_name: img_np})[0]
    output = output[0].transpose(1, 2, 0)  # (H, W, 3)
    output = ((output * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

    # Return PNG image
    output_image = Image.fromarray(output)
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

# Pad input to nearest multiple of N (e.g. 32)
def pad_to_multiple(img: Image.Image, multiple: int):
    width, height = img.size
    pad_w = (multiple - width % multiple) % multiple
    pad_h = (multiple - height % multiple) % multiple
    new_w = width + pad_w
    new_h = height + pad_h

    # Create blank image and paste original
    padded = Image.new("RGB", (new_w, new_h))
    padded.paste(img, (0, 0))

    for x in range(width, new_w):
        for y in range(height):
            padded.putpixel((x, y), img.getpixel((2 * width - x - 1, y)))

    for x in range(new_w):
        for y in range(height, new_h):
            padded.putpixel((x, y), padded.getpixel((x, 2 * height - y - 1)))

    return padded

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
