import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()


def embed_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        pooled_output = vision_outputs.pooler_output
        features = model.visual_projection(pooled_output)

    return features / features.norm(dim=-1, keepdim=True)


def cosine_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    return float((emb_a @ emb_b.T).item())