
import gradio as gr
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download

DEVICE      = torch.device("cpu")
NUM_CLASSES = 10
IMAGE_SIZE  = 512

CLASS_NAMES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]

CLASS_COLORS = np.array([
    [34,  139, 34 ], [0,   200, 0  ], [210, 180, 140],
    [139, 90,  43 ], [128, 128, 128], [255, 105, 180],
    [101, 67,  33 ], [169, 169, 169], [210, 180, 100],
    [135, 206, 235],
], dtype=np.uint8)

def get_model():
    return smp.DeepLabV3Plus(
        encoder_name="resnet50", encoder_weights=None,
        in_channels=3, classes=10, activation=None,
    )

model_path = hf_hub_download(
    repo_id="Aadarshhhhhhhh/desert-segmentation",
    filename="deeplabv3_desert_best.pth",
    repo_type="space",
)
model = get_model()
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
# Full checkpoint tha — model_state key se load karo
if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)
model.eval()

transform = A.Compose([
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    A.pytorch.ToTensorV2(),
])

def predict(input_image):
    image = np.array(
        Image.fromarray(input_image).resize((512,512), Image.BILINEAR),
        dtype=np.uint8
    )
    tensor = transform(image=image)["image"].unsqueeze(0)
    with torch.no_grad():
        pred = model(tensor).argmax(dim=1).squeeze(0).numpy()
    color_mask = CLASS_COLORS[pred]
    overlay    = (image * 0.6 + color_mask * 0.4).astype(np.uint8)
    total = pred.size
    info  = "**Class Distribution:**\n"
    for i, name in enumerate(CLASS_NAMES):
        pct = (pred == i).sum() / total * 100
        if pct > 0.5:
            info += f"`{name:<16}` {pct:5.1f}%  {'█' * int(pct/3)}\n"
    return Image.fromarray(color_mask), Image.fromarray(overlay), info

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=[
        gr.Image(type="pil", label="Segmentation Mask"),
        gr.Image(type="pil", label="Overlay"),
        gr.Markdown(),
    ],
    title="Desert Segmentation",
)
demo.launch()
