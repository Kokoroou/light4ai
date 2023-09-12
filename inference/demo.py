import os
from pathlib import Path

if __name__ == "__main__":
    current_dir = Path(__file__).parent.resolve()
    demo_path = current_dir / "demo" / "main.py"

    os.system(f"streamlit run {demo_path}")
