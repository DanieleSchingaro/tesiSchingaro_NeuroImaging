#tests/test_ldm_dataset.py
"""
Test rapido di ldm_dataset.py con .npz fittizi.
Verifica caricamento, padding condizionale e batching senza aver bisogno
del VAE trainato o degli embedding reali.
Esegui con: python3 -m tests.test_ldm_dataset
"""

import os
import json
import tempfile
import shutil
import numpy as np 
import torch
from src.data.ldm_dataset import(
    pad_to_divisible,
    LatentDataset,
    create_ldm_dataloader,
)

def test_pad_already_divisible():
    """
    Latente [4,64,64,64] già divisibile per 8 -> no padding.
    """
    latent=torch.randn(4,64,64,64)
    out=pad_to_divisible(latent,k=8)
    assert out.shape==(4,64,64,64), f"Shape cambiata: {out.shape}"
    print("OK pad_to_divisible: shape già divisibile non viene toccata")

def test_pad_not_divisible():
    """
    Latente [4,45,54,45] -> padding a [4,48,56,48]
    """
    latent=torch.randn(4,45,54,45)
    out=pad_to_divisible(latent, k=8)
    assert out.shape==(4,48,56,48), f"Padding errato: {out.shape}"
    #verifica che il padding sia zero
    assert out[:,45:,:,:].abs().sum()==0, "Padding non azzerato (X)"
    print("OK pad_to_divisible: [4,45,54,45] -> [4,48,56,48] con zero pad")

def test_dataset_and_loader():
    """
    Crea .npz fittizi, verifica Dataset e DataLoader
    """
    tmp_dir=tempfile.mkdtemp()
    try:
        data_list=[]
        for i in range(6):
            z=np.random.randn(4,64,64,64).astype(np.float32)
            affine=np.eye(4, dtype=np.float32)
            path=os.path.join(tmp_dir, f"vol_{i}_emb.npz")
            np.savez(path, z=z, affine=affine)
            data_list.append({"image": path, "source": "test"})
        
        #Dataset
        ds=LatentDataset(data_list, k=8)
        assert len(ds)==6, f"Len errata: {len(ds)}"
        sample=ds[0]
        assert sample["latent"].shape==(4,64,64,64), f"Shape: {sample['latent'].shape}"
        assert sample["latent"].dtype==torch.float32
        assert sample["source"]=="test"
        print("OK LatentDataset: caricamento e shape corretti")

        #DataLoader
        loader=create_ldm_dataloader(
            data_list, batch_size=2, is_train=True, num_workers=0, k=8,
        )
        batch=next(iter(loader))
        assert batch["latent"].shape==(2,4,64,64,64), f"Batch shape: {batch['latent'].shape}"
        print("OK create_ldm_dataloader: batching [2,4,64,64,64] corretto")

    finally:
        shutil.rmtree(tmp_dir)

def main():
    print("="*50)
    print("TEST ldm_dataset.py (con .npz fittizi)")
    print("="*50)
    test_pad_already_divisible()
    test_pad_not_divisible()
    test_dataset_and_loader()
    print("="*50)
    print("Tutti i test superati")
    print("="*50)

if __name__=="__main__":
    main()