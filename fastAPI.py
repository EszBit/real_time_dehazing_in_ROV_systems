from fastapi import FastAPI, File, UploadFile
import uvicorn
import onnxruntime as ort
from PIL import Image
import numpy as np
import io
from fastapi.responses import Response

app = FastAPI()

# Load ONNX model with MPS support
sess_options = ort.SessionOptions()
session = ort.InferenceSession(
    "funiegan_model_dynamic.onnx",
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
    sess_options=sess_options
)
input_name = session.get_inputs()[0].name

print("Available providers:", ort.get_available_providers())
print("Active provider(s):", session.get_providers())

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img = pad_to_multiple(img, 32)

    img_np = (np.array(img).astype(np.float32) / 255.0 - 0.5) / 0.5
    img_np = np.transpose(img_np, (2, 0, 1))[None, :, :, :]

    output = session.run(None, {input_name: img_np})[0]
    output = output[0].transpose(1, 2, 0)
    output = ((output * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

    output_image = Image.fromarray(output)
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

def pad_to_multiple(img: Image.Image, multiple: int):
    width, height = img.size
    pad_w = (multiple - width % multiple) % multiple
    pad_h = (multiple - height % multiple) % multiple
    new_w = width + pad_w
    new_h = height + pad_h
    padded = Image.new("RGB", (new_w, new_h))
    padded.paste(img, (0, 0))
    for x in range(width, new_w):
        for y in range(height):
            padded.putpixel((x, y), img.getpixel((2 * width - x - 1, y)))
    for x in range(new_w):
        for y in range(height, new_h):
            padded.putpixel((x, y), padded.getpixel((x, 2 * height - y - 1)))
    return padded

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
