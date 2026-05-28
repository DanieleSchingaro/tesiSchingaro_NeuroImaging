"""
Dataset e DataLoader per MRI cerebrali T1 skull-stripped.
Le transforms sono in src/data/transforms.py.
"""

import os
import random
import json
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from monai.data import CacheDataset
from src.data.transforms import get_vae_transforms


# Cartelle del dataset HC
DATASET_DIRS=[
    "data/raw/hc_adni_brain_mask",
    "data/raw/hc_nifd_brain_mask",
    "data/raw/hc_oasis1_brain_mask",
    "data/raw/hc_oasis2_brain_mask",
    "data/raw/hc_oasis3_brain_mask",
    "data/raw/hc_ppmi_brain_mask",
]


def get_file_list(dataset_dirs: list[str])->list[dict]:
    """
    Scansiona le cartelle del dataset e restituisce
    una lista di dizionari {image: path} compatibile con MONAI.
    """
    files=[]
    for folder in dataset_dirs:
        folder=Path(folder)
        if not folder.exists():
            print(f"Cartella non trovata: {folder}")
            continue
        nii_files=sorted(folder.glob("*.nii.gz"))
        for f in nii_files:
            files.append({"image": str(f)})
    print(f"Trovati {len(files)} file totali")
    return files


def split_dataset(
    files:list[dict],
    train_ratio:float=0.8,
    val_ratio:float=0.1,
    seed:int=42,
)->tuple[list, list, list]:
    """
    Divide il dataset in Train/Val/Test in modo riproducibile.
    80% train, 10% val, 10% test.
    """
    random.seed(seed)
    files_shuffled=files.copy()
    random.shuffle(files_shuffled)
    n=len(files_shuffled)
    n_train=int(n * train_ratio)
    n_val=int(n * val_ratio)
    train_files=files_shuffled[:n_train]
    val_files=files_shuffled[n_train:n_train + n_val]
    test_files=files_shuffled[n_train + n_val:]
    print(f"Split dataset: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    return train_files, val_files, test_files


def save_split(
    train_files:list,
    val_files:list,
    test_files:list,
    output_path:str="data/splits/dataset.json",
)->None:
    """Salva gli split in un JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    splits={
        "training": train_files,
        "validation": val_files,
        "test": test_files,
    }
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Split salvati in {output_path}")


def load_splits(splits_path: str)->tuple[list, list, list]:
    """Carica gli split da un file JSON esistente."""
    with open(splits_path, "r") as f:
        splits=json.load(f)
    return splits["training"], splits["validation"], splits["test"]


def create_dataloader(
    files:list[dict],
    patch_size:tuple=(64, 64, 64),
    val_patch_size:Optional[tuple]=None,
    batch_size:int=1,
    is_train:bool=True,
    random_aug:bool=True,
    cache_rate:float=0.5,
    num_workers:int=4,
    k:int=4,
    sampler=None,
)->DataLoader:
    """
    Crea un DataLoader MONAI con CacheDataset.
    Supporta DistributedSampler per training multi-GPU con DDP.
    """
    transforms=get_vae_transforms(
        patch_size=patch_size,
        val_patch_size=val_patch_size,
        is_train=is_train,
        random_aug=random_aug,
        k=k,
    )
    dataset=CacheDataset(
        data=files,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    # shuffle solo se non usiamo sampler DDP
    shuffle_flag=is_train and sampler is None
    dataloader=DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True,
    )
    return dataloader


def setup_dataloaders(
    config:dict,
    splits_path:str="data/splits/dataset.json",
)->tuple:
    """
    Crea o carica gli split e restituisce i tre DataLoader.
    Gestisce automaticamente DDP con DistributedSampler.
    """
    if os.path.exists(splits_path):
        print(f"Splits esistenti caricati da {splits_path}")
        train_files, val_files, test_files=load_splits(splits_path)
    else:
        print("Creazione nuovi splits")
        files=get_file_list(DATASET_DIRS)
        train_files, val_files, test_files=split_dataset(files)
        save_split(train_files, val_files, test_files, splits_path)

    patch_size=tuple(config.get("patch_size", [64, 64, 64]))
    val_patch_size=config.get("val_patch_size", None)
    if val_patch_size is not None:
        val_patch_size=tuple(val_patch_size)
    batch_size=config.get("batch_size", 1)
    cache_rate=config.get("cache_rate", 0.0)
    num_workers=config.get("num_workers", 4)
    random_aug=config.get("random_aug", True)

    is_distributed=dist.is_available() and dist.is_initialized()
    rank=dist.get_rank() if is_distributed else 0
    world_size=dist.get_world_size() if is_distributed else 1

    train_sampler=DistributedSampler(
        train_files, num_replicas=world_size, rank=rank, shuffle=True, seed=42,
    ) if is_distributed else None

    val_sampler=DistributedSampler(
        val_files, num_replicas=world_size, rank=rank, shuffle=False,
    ) if is_distributed else None

    test_sampler=DistributedSampler(
        test_files, num_replicas=world_size, rank=rank, shuffle=False,
    ) if is_distributed else None

    train_loader=create_dataloader(
        train_files, patch_size=patch_size, val_patch_size=None,
        batch_size=batch_size, is_train=True, random_aug=random_aug,
        cache_rate=cache_rate, num_workers=num_workers, sampler=train_sampler,
    )
    val_loader=create_dataloader(
        val_files, patch_size=patch_size, val_patch_size=val_patch_size,
        batch_size=batch_size, is_train=False, random_aug=False,
        cache_rate=0.0, num_workers=num_workers, sampler=val_sampler,
    )
    test_loader=create_dataloader(
        test_files, patch_size=patch_size, val_patch_size=val_patch_size,
        batch_size=batch_size, is_train=False, random_aug=False,
        cache_rate=0.0, num_workers=num_workers, sampler=test_sampler,
    )

    return train_loader, val_loader, test_loader