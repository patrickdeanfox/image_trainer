import os
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load the model (using float16 to save VRAM)
model_id = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16
).to("cuda")

TRIGGER_WORD = "ohwx person"
input_dir = Path(os.path.expanduser("~/Apps/image_trainer/training_data/processed"))

count = 0
for img_path in sorted(input_dir.glob("*.png")):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)
    out = model.generate(**inputs, max_new_tokens=75)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    # Prepend trigger word
    full_caption = f"{TRIGGER_WORD}, {caption}"
    
    caption_file = img_path.with_suffix(".txt")
    caption_file.write_text(full_caption)
    print(f"Captioned {img_path.name}: {full_caption}")
    count += 1

print(f"\nDone! {count} text files created.")
