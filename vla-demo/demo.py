from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

actions = ["do nothing", "water soil", "pick up seed", "move arm to soil sensor"]

def vla_action_from_image(image_path):
    image = Image.open(image_path)
    inputs = processor(text=actions, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    action_index = probs.argmax().item()
    action = actions[action_index]
    print(f"[VLA] Recommended action: {action}")
    return action

def execute_action(action):
    print(f"[Robot] Executing: {action}\n")

if __name__ == "__main__":
    image_path = "image.png"
    action = vla_action_from_image(image_path)
    execute_action(action)
