import os
import torchvision.datasets as datasets

def download_voc(root: str = "data"):
    print("Downloading PASCAL VOC 2012 — Segmentation split...")
    print("Size: ~2GB. This will take a few minutes.\n")

    # torchvision handles download + extraction automatically
    for split in ["train", "val"]:
        datasets.VOCSegmentation(
            root=root,
            year="2012",
            image_set=split,
            download=True
        )
        print(f"✅ {split} split ready.")

    # Verify structure
    expected_dirs = [
        os.path.join(root, "VOCdevkit", "VOC2012", "JPEGImages"),
        os.path.join(root, "VOCdevkit", "VOC2012", "SegmentationClass"),
        os.path.join(root, "VOCdevkit", "VOC2012", "ImageSets", "Segmentation"),
    ]

    print("\nVerifying structure...")
    all_ok = True
    for d in expected_dirs:
        exists = os.path.isdir(d)
        status = "✅" if exists else "❌"
        print(f"  {status} {d}")
        if not exists:
            all_ok = False

    if all_ok:
        # Count files
        imgs = len(os.listdir(expected_dirs[0]))
        masks = len(os.listdir(expected_dirs[1]))
        print(f"\n✅ Download complete.")
        print(f"   Total JPEGImages : {imgs}")
        print(f"   Total Masks      : {masks}")
    else:
        print("\n❌ Some directories are missing. Re-run this script.")

if __name__ == "__main__":
    download_voc(root="data")