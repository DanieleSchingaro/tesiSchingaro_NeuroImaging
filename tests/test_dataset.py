"""
Test rapido del DataLoader.
Esegui con: python3 -m tests.test_dataset
"""
from src.data.dataset import get_file_list, DATASET_DIRS, split_dataset, save_split, create_dataloader

def test_dataloader():
    config = {
        "patch_size": [64, 64, 64],
        "val_patch_size": None,
        "batch_size": 1,
        "cache_rate": 0.0,
        "num_workers": 2,
        "random_aug": False,
    }

    files = get_file_list(DATASET_DIRS)
    train_files, val_files, test_files = split_dataset(files)
    save_split(train_files, val_files, test_files)

    train_loader = create_dataloader(
        train_files,
        patch_size=tuple(config["patch_size"]),
        batch_size=config["batch_size"],
        is_train=True,
        random_aug=config["random_aug"],
        cache_rate=config["cache_rate"],
        num_workers=config["num_workers"],
    )

    batch = next(iter(train_loader))
    images = batch["image"]
    print(f"Batch shape: {images.shape}")
    print(f"Min: {images.min():.4f}, Max: {images.max():.4f}")
    print(f"Dtype: {images.dtype}")
    assert images.shape == (1, 1, 64, 64, 64), "Shape non corretta!"
    assert images.min() >= 0.0, "Min fuori range!"
    assert images.max() <= 1.0, "Max fuori range!"
    print("Test DataLoader superato!")

if __name__ == "__main__":
    test_dataloader()
