"""One-time helper: extract one CIFAR-10 test-set image per class and
save as PNG into this directory.

Run after `pip install -r app/requirements.txt`:

    cd app/samples
    python make_samples.py
"""
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def main():
    try:
        import datasets
    except ImportError:
        raise SystemExit("`datasets` not installed. Run: pip install datasets")

    print("Downloading CIFAR-10 test split via Hugging Face...")
    ds = datasets.load_dataset("cifar10", split="test")
    ds.set_format("numpy")
    images = np.array(ds["img"])
    labels = np.array(ds["label"])

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(exist_ok=True)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        # First test-set image of this class
        idx = int(np.where(labels == cls_idx)[0][0])
        img = Image.fromarray(images[idx])
        out_path = out_dir / f"{cls_idx:02d}_{cls_name}.png"
        img.save(out_path)
        print(f"  ✓ {out_path.name}")

    print(f"\nDone. {len(CLASS_NAMES)} sample images saved to {out_dir}/")


if __name__ == "__main__":
    main()
