import torch.nn.functional as F

def pad_to_multiple(tensor, multiple=32):
    # tensor shape: (C, H, W)
    _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')


# Step 1: Get frame from UScanner
frame = get_next_frame_from_scanner()

# Step 2: Convert to tensor & normalize
input_tensor = transform_to_tensor(frame)  # torch.Tensor with shape [C, H, W]

# Step 3: Resize or pad
input_tensor = pad_to_multiple(input_tensor, multiple=32)

# Step 4: Inference
input_tensor = input_tensor.unsqueeze(0).to(device)
output_tensor = model(input_tensor)
