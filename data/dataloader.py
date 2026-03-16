from torch.utils.data import DataLoader
from data.dataset import VOCSegmentationDataset


def get_dataloader(
    root: str = "data",
    split: str = "train",
    image_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 0,       # Keep 0 on M1
    ignore_index: int = 255
) -> DataLoader:

    dataset = VOCSegmentationDataset(
        root=root,
        split=split,
        image_size=image_size,
        ignore_index=ignore_index
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=False,       # pin_memory=False for MPS
        drop_last=(split == "train")
    )

    return loader