from pathlib import Path

from PIL import Image

if __name__ == "__main__":
    from models import Model

    model = Model(model_name="tiny_vit_21m_224", task="classify")

    current_dir = Path(__file__).parent.resolve()
    image_path = current_dir / "models" / "classify" / "TinyViT" / ".figure" / "cat.jpg"
    image = Image.open(image_path)

    model(image)
