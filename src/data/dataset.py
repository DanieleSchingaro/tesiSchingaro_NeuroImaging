#src/data/dataset.py

"""
Dataset e DataLoader per MRI cerebrali
Caricamento immagini NIfTI da multiple cartelle e applicazione transforms MONAI
"""

import os
import json
from pathlib import Path
from typing import Optional
import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import(
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
)

#Cartelle del dataset HC
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
    una lista di dizionari {immagine:path} compatibile con MONAI
    """
    files=[]
    for folder in dataset_dirs:
        folder=Path(folder)
        if not folder.exists():
            print(f"Cartella non trovata: {folder}")
            continue
        #raccoglie i file .nii.gz nella cartella
        nii_files=sorted(folder.glob("*.nii.gz"))
        for f in nii_files:
            files.append({"immagine": str(f)})
    print(f"Trovati {len(files)} file totali")
    return files

def get_transforms(patch_size:tuple=(64,64,64), is_train:bool=True)->Compose:
    """
    Transforms MONAI per training e validazione.
    Per training:
    -Crop casuale a patch 64x64x64 
    -Flip e rotazioni casuali per augmentation
    Per la validazione:
    -Solo normalizzazione
    """
    base_transforms=[
        #caricamento file NIfTI da path
        LoadImaged(keys=["image"]),
        #impostiamo formato (C,H,W,D)
        EnsureChannelFirstd(keys=["image"]),
        #orientamento RAS
        Orientationd(keys=["image"], axcodes="RAS"),
        #normalizzazione intensità tra 0 e 1
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=0.5,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        #conversione in tensor float32
        EnsureTyped(keys=["image"], dtype=torch.float32)
    ]

    if is_train:
        #augmentation
        train_transforms=[
            #crop casuale
            RandSpatialCropd(
                keys=["image"],
                roi_size=patch_size,
                random_size=False,
            ),
            #flip casuale lungo i 3 assi
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            #rotazione casuale di 90 gradi
            RandRotate90d(keys=["image"], prob=0.5, max_k=3),
        ]
        return Compose(base_transforms+train_transforms)
    
    #per validation solo base_transforms
    return Compose(base_transforms)

def split_dataset(files:list[dict], train_ratio:float=0.8, val_ratio:float=0.1, seed:int=42,)-> tuple[list, list, list]:
    """
    Divide il dataset in Train/Val/Test in modo riproducibile.
    80% train, 10%val, 10%test.
    """
    import random
    random.seed(seed)

    files_shuffled=files.copy()
    random.shuffle(files_shuffled)

    n=len(files_shuffled)
    n_train=int(n*train_ratio)
    n_val=int(n*val_ratio)

    train_files=files_shuffled[:n_train]
    val_files=files_shuffled[n_train:n_train+n_val]
    test_files=files_shuffled[n_train+n_val:]

    print(f"Split dataset: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    return train_files, val_files, test_files

def save_split(train_files:list, val_files:list, test_files:list, output_path:str="data/splits/dataset.json",)->None:
    """
    Salva gli split in un json.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    splits={
        "training":train_files,
        "validation":val_files,
        "test":test_files,
    }
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Split salvati in {output_path}")

def load_splits(splits_path:str)->tuple[list,list,list]:
    """
    Carica gli split da un file json esistente.
    """
    with open(splits_path, "r") as f:
        splits=json.load(f)
    return splits["training"], splits["validation"], splits["testing"]

def create_dataloader(files:list[dict], patch_size:tuple=(64,64,64), batch_size:int=1, is_train:bool=True, cache_rate:float=0.5, num_workers:int=4,)->DataLoader:
    """
    Creazione DataLoader MONAI con CacheDataset.
    CacheDataset carica e preprocessa i dati una sola volta accelerando il training
    """
    transforms=get_transforms(patch_size=patch_size, is_train=is_train)
    #cache_rate=0 -> nessuna cache
    #cache_rate=1 -> tutto in memoria
    dataset=CacheDataset(
        data=files,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    #shuffle per il training
    dataloader=DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True, #accelera trasferimento CPU->GPU
        persistent_workers=True if num_workers>0 else False,
    )

    return dataloader

def setup_dataloaders(config:dict, splits_path:str="data/splits/dataset.json",)->tuple[DataLoader, DataLoader, DataLoader]:
    """
    Raccoglie i file, crea o carica gli splits e restituisce i tre DataLoader
    """
    #controllo esistenza splits
    if os.path.exists(splits_path):
        print(f"Splits esistenti caricati da {splits_path}")
        train_files, val_files, test_files=load_splits(splits_path)
    else:
        print(f"Creazione nuovi splits")
        files=get_file_list(DATASET_DIRS)
        train_files, val_files, test_files=split_dataset(files)
        save_split(train_files, val_files, test_files, splits_path)

    #parametri del config
    patch_size=tuple(config.get("patch_size", [64,64,64]))
    batch_size=config.get("batch_size", 1)
    cache_rate=config.get("cache_rate", 0.5)
    num_workers=config.get("num_workers", 4)

    #creazione dataloader
    train_loader=create_dataloader(
        train_files, patch_size, batch_size, is_train=True, cache_rate=cache_rate, num_workers=num_workers,
    )
    val_loader=create_dataloader(
        val_files, patch_size, batch_size, is_train=False, cache_rate=0.0, num_workers=num_workers,
    )
    test_loader=create_dataloader(
        test_files, patch_size, batch_size, is_train=False, cache_rate=0.0, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader

if __name__ =="__main__":
    """
    Test DataLoader
    Esecuzione: python3 -m src.data.dataset
    Verifica del corretto caricamento delle immagini
    """
    config={
        "patch_size":[64,64,64],
        "batch_size":1,
        "cache_rate":0.0,
        "num_workers":2,
    }

    #raccolta file e creazione split
    files=get_file_list(DATASET_DIRS)
    train_files, val_files, test_files=split_dataset(files)
    save_split(train_files, val_files, test_files)

    #creazione dataloader di training
    train_loader=create_dataloader(
        train_files,
        patch_size=tuple(config["patch_size"]),
        batch_size=config["batch_size"],
        is_train=True,
        cache_rate=config["cache_rate"],
        num_workers=config["num_workers"],
    )  

    #primo batch e sue dimensioni
    batch=next(iter(train_loader))
    images=batch["image"]
    print(f"Batch shape: {images.shape}")
    print(f"Min: {images.min():.4f}, Max: {images.max():.4f}")
    print(f"Dtype: {images.dtype}")
    print(f"Funzionamento corretto del DataLoader")