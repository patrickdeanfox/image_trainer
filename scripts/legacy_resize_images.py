from PIL import Image
from pathlib import Path
import os

# Adjusted to your Apps/image_trainer path
input_dir = Path(os.path.expanduser("~/Apps/image_trainer/training_data/raw"))
output_dir = Path(os.path.expanduser("~/Apps/image_trainer/training_data/processed"))
output_dir.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 1024

count = 0
for i, img_path in enumerate(sorted(input_dir.glob("*"))):
    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            # Resize smallest side to 1024
            scale = TARGET_SIZE / min(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            
            # Center Crop
            w, h = img.size
            left = (w - TARGET_SIZE) // 2
            top = (h - TARGET_SIZE) // 2
            img = img.crop((left, top, left + TARGET_SIZE, top + TARGET_SIZE))
            
            img.save(output_dir / f"{i:04d}.png")
            print(f"Processed: {img_path.name} -> {i:04d}.png")
            count += 1
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

print(f"\nDone! {count} images ready in ~/Apps/image_trainer/training_data/processed")
