#tests/check_best_model_vae.py

"""
Verifica il checkpoint del miglior VAE e le metriche salvate su MLFlow.
Esegui con: python3 tests/check_best_model_vae.py
"""
import torch
import mlflow 
from pathlib import Path 

#Checkpoint
checkpoint_path=Path("outputs/models/autoencoder_best.py")
if not checkpoint_path.exists():
    print(f"File non trovato: {checkpoint_path}")
    exit(1)

print(f"Caricamento checkpoint: {checkpoint_path}")
checkpoint=torch.load(checkpoint_path, map_location="cpu")

print("\n===INFO CHECKPOINT===")
if "epoch" in checkpoint:
    print(f"Epoca salvata: {checkpoint['epoch']}")
if "val_loss" in checkpoint:
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")

print("\n===CHIAVI PRESENTI===")
for key in checkpoint.keys():
    print(f"-{key}")

if "autoencoder_state_dict" in checkpoint:
    total_params=sum(
        v.numel() for v in checkpoint["autoencoder_state_dict"].values()
    )
    print(f"\nParametri autoencoder: {total_params:,}")

if "discriminator_state_dict" in checkpoint:
    total_params_d=sum(
        v.numel() for v in checkpoint["discriminator_state_dict"].values()
    )
    print(f"\nParametri discriminator: {total_params_d:,}")

#MLFlow
print("\n===METRICHE MLFLOW===")
try:
    client=mlflow.tracking.MlflowClient()
    experiments=client.search_experiments(filter_string="name='VAE_training'")
    if experiments:
        runs=client.search_runs(
            experiment_ids=[experiments[0].experiment_id],
            order_by=["start_time DESC"]
        )
        if runs:
            #ultimo run
            run=runs[0]
            print(f"Run ID: {run.info.run_id}")
            print(f"Run name: {run.info.run_name}")
            print(f"Status: {run.info.status}")
            print(f"\nMetriche:")
            for key, value in sorted(run.data.metrics.items()):
                print(f"{key:<25}: {value:.6f}")
            print(f"\nParametri:")
            for key, value in sorted(run.data.params.items()):
                print(f"{key:<25}: {value}")
        else:
            print("Nessun run trovato")
    else:
        print("Esperimento VAE_training non trovato")
except Exception as e:
    print(f"Errore MLFlow: {e}")