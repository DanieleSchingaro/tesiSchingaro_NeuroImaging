#src/training/train_vae.py
"""
Training del VAE per MRI cerebrali T1 skull-stripped
Basato su NV-Generate-CTMR (NVIDIA, 2026)
"""

import os
import json
import time
import torch
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from generative.networks.nets import PatchDiscriminator
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.losses import PerceptualLoss
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.utils import set_determinism
from src.data.dataset import setup_dataloaders
import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist
from contextlib import nullcontext

def load_config(config_path:str)->dict:
    """Carica file di configurazione json"""
    with open(config_path, "r") as f:
        return json.load(f)

def setup_ddp()->tuple[int, torch.device]:
    """
    Inizializza DDP con torchrun.
    torchrun setta LOCAL_RANK, RANK, WORLD_SIZE automaticamente.
    """
    dist.init_process_group(backend="nccl")
    local_rank=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device=torch.device(f"cuda:{local_rank}")
    return local_rank, device

def setup_models(config:dict, device:torch.device):
    """
    Inizializza VAE e discriminatore.
    AutoencoderKL: encoder-decoder nello spazio latente.
    PatchDiscriminator: discriminatore per la loss.
    """

    autoencoder=AutoencoderKlMaisi(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        num_channels=(64, 128, 256),
        num_res_blocks=(2, 2, 2),
        norm_num_groups=32,
        norm_eps=1e-6,
        attention_levels=(False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=True,   # ← gradient checkpointing per 256³
        num_splits=4,             # ← split per efficienza memoria
        dim_split=1,
    )

    discriminator=PatchDiscriminator(
        spatial_dims=3,
        num_layers_d=3,
        num_channels=32,
        in_channels=1,
        out_channels=1,
    )

    autoencoder=autoencoder.to(device)
    discriminator=discriminator.to(device)

    local_rank=int(os.environ.get("LOCAL_RANK", 0))

    if dist.is_available() and dist.is_initialized():
        print(f"[rank{local_rank}] Uso DDP")
        autoencoder=torch.nn.parallel.DistributedDataParallel(
            autoencoder,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        discriminator=torch.nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        print(f"Uso singola GPU/CPU")

    return autoencoder, discriminator

def setup_losses(device:torch.device):
    """
    Inizializza le loss functions:
    L1Loss
    PerceptualLoss
    PatchAdversarialLoss
    """
    #Loss L1
    l1_loss=L1Loss()

    #PerceptualLoss
    perceptual_loss=PerceptualLoss(
        spatial_dims=3,
        network_type="squeeze",
        is_fake_3d=True,
        fake_3d_ratio=0.25,
    ).to(device)

    #Adversarial Loss
    adv_loss=PatchAdversarialLoss(criterion="least_squares")

    return l1_loss, perceptual_loss, adv_loss

def setup_optimizers(autoencoder:torch.nn.Module, discriminator:torch.nn.Module, lr:float):
    """
    Inizializza gli ottimizzatori Adam per generator e discriminator.
    Due ottimizzatori seperati --> training stabile
    """
    #ottimizzatore per l'Autoencoder
    optimizer_g=torch.optim.Adam(
        params=autoencoder.parameters(),
        lr=lr,
    )

    #ottimizzatore per il discriminator
    optimizer_d=torch.optim.Adam(
        params=discriminator.parameters(),
        lr=lr*0.5,
    )

    return optimizer_g, optimizer_d

def train_one_epoch(
    epoch,
    autoencoder,
    discriminator,
    train_loader,
    optimizer_g,
    optimizer_d,
    l1_loss,
    perceptual_loss,
    adv_loss,
    scaler_g,
    device,
    kl_weight,
    perceptual_weight,
    adv_weight,
    warm_up_epochs,
):

    autoencoder.train()
    discriminator.train()

    epoch_recon_loss=0.0
    epoch_gen_loss=0.0
    epoch_disc_loss=0.0
    epoch_kl_loss=0.0

    #ranngo per barra tqdm --> solo rank 0 == solo una barra
    rank=dist.get_rank() if dist.is_initialized() else 0

    progress_bar=tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc=f"Epoch {epoch}",
                        ncols=100,
                        disable=rank!=0,)

    for step, batch in progress_bar:
        images=batch["image"].to(device)
        optimizer_g.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=True):
            reconstruction, z_mu, z_sigma=autoencoder(images)
            recon_loss=l1_loss(reconstruction.float(), images.float())

            eps=1e-10
            kl_loss=0.5*(
                z_mu.pow(2)
                + z_sigma.pow(2)
                - torch.log(z_sigma.pow(2) + eps)
                - 1
            )

            kl_loss=kl_loss.mean()

            p_loss=perceptual_loss(reconstruction.float(), images.float())

            loss_g=recon_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)

            if epoch>=warm_up_epochs:
                fake_in=reconstruction.float().contiguous()
                logits_fake=discriminator(fake_in)[-1]
                generator_loss=adv_loss(
                    logits_fake,
                    target_is_real=True,
                    for_discriminator=False,
                )
                loss_g +=adv_weight*generator_loss

                if not torch.isnan(generator_loss):
                    epoch_gen_loss+=generator_loss.item()

        if torch.isnan(loss_g) or torch.isinf(loss_g):
            optimizer_g.zero_grad(set_to_none=True)
            continue

        scaler_g.scale(loss_g).backward()
        scaler_g.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
        scaler_g.step(optimizer_g)
        scaler_g.update()

        if epoch >= warm_up_epochs and step % 2==0:
            optimizer_d.zero_grad(set_to_none=True)

            # .detach().clone() invece di solo .detach()
            recon_detached=reconstruction.detach().clone().float()
            images_detached=images.detach().clone().float()

            logits_fake=discriminator(recon_detached.contiguous())[-1]
            logits_fake=torch.clamp(logits_fake, -10, 10)
            loss_d_fake=adv_loss(logits_fake, target_is_real=False, for_discriminator=True)

            logits_real=discriminator(images_detached.contiguous())[-1]
            logits_real=torch.clamp(logits_real, -10, 10)
            loss_d_real=adv_loss(logits_real, target_is_real=True, for_discriminator=True)

            discriminator_loss=(loss_d_fake + loss_d_real)*0.5
            loss_d=adv_weight*discriminator_loss

            if torch.isnan(loss_d) or torch.isinf(loss_d):
                print(f"[WARNING] NaN discriminator epoch {epoch} step {step}")
                optimizer_d.zero_grad(set_to_none=True)
                continue

            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(
                discriminator.module.parameters() if hasattr(discriminator, 'module')
                else discriminator.parameters(),
                max_norm=1.0
            )
            optimizer_d.step()
            epoch_disc_loss+=discriminator_loss.item()

        epoch_recon_loss+=recon_loss.item()
        epoch_kl_loss+=kl_loss.item()

        progress_bar.set_postfix({
            "recon":f"{epoch_recon_loss/(step+1):.4f}",
            "kl":f"{epoch_kl_loss/(step+1):.4f}",
            "disc":f"{epoch_disc_loss/(step+1):.4f}",
        })

    n_steps=len(train_loader)

    return {
        "recon_loss": epoch_recon_loss / n_steps,
        "kl_loss": epoch_kl_loss / n_steps,
        "gen_loss": epoch_gen_loss / n_steps,
        "disc_loss": epoch_disc_loss / n_steps,
    }

def validate(
    epoch:int,
    autoencoder:torch.nn.Module,
    val_loader,
    l1_loss,
    perceptual_loss,
    device:torch.device,
    perceptual_weight:float,
)->dict:
    """
    Validazione del VAE sul validation set.
    Calcola solo la loss di ricostruzione e la perceptual
    """
    autoencoder.eval()
    val_recon_loss=0.0
    val_p_loss=0.0

    #ranngo per barra tqdm --> solo rank 0 == solo una barra
    rank=dist.get_rank() if dist.is_initialized() else 0

    with torch.no_grad():
        for step,batch in enumerate(tqdm(
            val_loader,
            desc=f"Validation epoch {epoch}",
            ncols=100,
            disable=rank!=0,
        )):
            images=batch["image"].to(device)
            with autocast("cuda", enabled=True):
                #forward pass
                reconstruction, z_mu, z_sigma=autoencoder(images)
                #loss reconstruction
                recon_loss=l1_loss(reconstruction.float(), images.float())
                #perceptual loss
                p_loss=perceptual_loss(reconstruction.float(), images.float())

            val_recon_loss+=recon_loss.item()
            val_p_loss+=p_loss.item()
    
    n_steps=len(val_loader)
    return{
        "val_recon_loss": val_recon_loss/n_steps,
        "val_p_loss": val_p_loss/n_steps,
    }

def save_checkpoint(
    epoch:int,
    autoencoder:torch.nn.Module,
    discriminator:torch.nn.Module,
    optimizer_g:torch.optim.Optimizer,
    optimizer_d:torch.optim.Optimizer,
    val_loss:float,
    save_dir:str,
    is_best:bool=False,
)->None:
    """
    Salva il checkpoint del modello.
    Se is_best=True salva anche il miglior modello separatamente
    """
    os.makedirs(save_dir, exist_ok=True)

    #estrazione del modello da DataParallel se necessario
    autoencoder_state=(
        autoencoder.module.state_dict()
        if isinstance(autoencoder, torch.nn.parallel.DistributedDataParallel)
        else autoencoder.state_dict()
    )
    discriminator_state=(
        discriminator.module.state_dict()
        if isinstance(discriminator, torch.nn.parallel.DistributedDataParallel)
        else discriminator.state_dict()
    )

    #checkpoint
    checkpoint={
        "epoch":epoch,
        "autoencoder_state_dict":autoencoder_state,
        "discriminator_state_dict":discriminator_state,
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict":optimizer_d.state_dict(),
        "val_loss":val_loss,
    }

    #salvataggio ultimo checkpoint
    last_path=os.path.join(save_dir, "autoencoder_last.pt")
    torch.save(checkpoint, last_path)

    #salvataggio del miglior modello in maniera separata
    if is_best:
        best_path=os.path.join(save_dir, "autoencoder_best.pt")
        torch.save(checkpoint, best_path)
        print(f"Miglior modello salvato (val_loss={val_loss:.4f})")

def save_learning_curves(
    train_recon_loss:list,
    val_recon_loss:list,
    train_gen_loss:list,
    train_disc_loss:list,
    save_dir:str,
    val_interval:int,
    n_epochs:int,
)->str:
    """
    Salva il grafico delle learning curves, loggandolo su MLFlow
    3 subplot:
    -Loss ricostruita (train+val)
    -Generator loss
    -Discriminator loss
    """
    fig, axes=plt.subplots(1,3,figsize=(18,5))
    fig.suptitle("VAE Learning Curves", fontsize=16)

    epochs=np.arange(1, n_epochs+1)
    val_epochs=np.arange(val_interval, n_epochs+1, val_interval)

    #subplot 1 --> loss ricostruita
    axes[0].plot(epochs, train_recon_loss, color="C0", label="Train")
    if val_recon_loss:
        axes[0].plot(val_epochs[:len(val_recon_loss)], val_recon_loss, color="C1", label="Validation")
    
    axes[0].set_title("Loss Ricostruita (L1)")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    #subplot 2 --> generator loss
    axes[1].plot(epochs, train_gen_loss, color="C2", label="Generator")
    axes[1].set_title("Generator Loss (Adversarial)")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    #subplot 3 --> discriminator loss
    axes[2].plot(epochs, train_disc_loss, color="C3", label="Discriminator")
    axes[2].set_title("Discriminator Loss")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Loss")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()

    #salvataggio grafico
    os.makedirs(save_dir, exist_ok=True)
    curve_path=os.path.join("outputs", "metrics", "learning_curves_vae.png")
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Learning curves salvate in {curve_path}")
    return curve_path

def main():
    """
    Flusso:
    1)Carica la configurazione
    2)Inizializza modelli, loss e ottimizzatori
    3)Crea dataloaders
    4)Loop di training con MLFlow
    5)Salvataggio del migliore checkpoint
    """

    local_rank=int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device=torch.device(f"cuda:{local_rank}")

    #caricamento configurazioni
    config_vae=load_config("configs/config_vae.json")
    config_env=load_config("configs/environment.json")

    #parametri del training
    train_cfg=config_vae["autoencoder_train"]
    batch_size=train_cfg["batch_size"]
    patch_size=tuple(train_cfg["patch_size"])
    lr=train_cfg["lr"]
    n_epochs=train_cfg["n_epochs"]
    val_interval=train_cfg["val_interval"]
    kl_weight=train_cfg["kl_weight"]
    perceptual_weight=train_cfg["perceptual_weight"]
    adv_weight=train_cfg["adv_weight"]
    warm_up_epochs=train_cfg["warm_up_epochs"]
    cache_rate=train_cfg["cache_rate"]
    num_workers=train_cfg["num_workers"]

    #path
    save_dir=config_env["model_dir"]
    splits_path=config_env["json_data_list"]

    #setup device e seed
    local_rank, device=setup_ddp()
    device=torch.device(f"cuda:{local_rank}")
    print(f"Device: {device}")
    print(f"GPU disponibili: {torch.cuda.device_count()}")
    set_determinism(seed=42)

    rank = dist.is_initialized() and dist.get_rank() or 0
    is_main_process=rank==0

    if is_main_process:
        mlflow.set_experiment("VAE_training")
        mlflow.start_run(run_name=f"vae_epoch{n_epochs}_lr{lr}")

        mlflow.log_params({
            "lr": lr,
            "n_epoch": n_epochs,
            "batch_size": batch_size,
            "patch_size": str(patch_size),
            "kl_weight": kl_weight,
            "perceptual_weight": perceptual_weight,
            "adv_weight": adv_weight,
            "warm_up_epochs": warm_up_epochs,
            "n_gpus": torch.cuda.device_count(),
        })

        mlflow_run=None
    else:
        mlflow_run=nullcontext()

    with mlflow_run if mlflow_run is not None else nullcontext():

        #inizializzazione modelli
        autoencoder, discriminator=setup_models(config_vae, device)

        autoencoder=torch.nn.parallel.DistributedDataParallel(
            autoencoder,
            device_ids=[local_rank],
            output_device=local_rank
        )

        discriminator=torch.nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[local_rank],
            output_device=local_rank
        )

        #caricamento checkpoint per fine-tuning se finetune=True
        if config_env.get("finetune", False):
            ckpt_path=config_env.get("trained_autoencoder_path")
            if ckpt_path and os.path.exists(ckpt_path):
                checkpoint=torch.load(ckpt_path, map_location=device)
                autoencoder.module.load_state_dict(checkpoint["autoencoder_state_dict"])
                print(f"Checkpoint caricato da {ckpt_path}")

        #loss
        l1_loss, perceptual_loss, adv_loss=setup_losses(device)

        #optimizer
        optimizer_g, optimizer_d=setup_optimizers(autoencoder, discriminator, lr)

        scheduler_g=CosineAnnealingLR(
            optimizer_g, T_max=n_epochs, eta_min=1e-6
        )

        scaler_g=GradScaler("cuda")

        all_train_recon=[]
        all_val_recon=[]
        all_train_gen=[]
        all_train_disc=[]

        config_data = {
            "patch_size": list(patch_size),
            "batch_size": batch_size,
            "cache_rate": cache_rate,
            "num_workers": num_workers,
            "random_aug": config_vae["data_option"]["random_aug"]
        }

        train_loader, val_loader, _=setup_dataloaders(
            config=config_data,
            splits_path=splits_path,
        )

        print(f"Train: {len(train_loader.dataset)} immagini")
        print(f"Val: {len(val_loader.dataset)} immagini")

        best_val_loss=float("inf")
        total_start=time.time()

        for epoch in range(n_epochs):

            train_metrics=train_one_epoch(
                epoch=epoch,
                autoencoder=autoencoder,
                discriminator=discriminator,
                train_loader=train_loader,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                l1_loss=l1_loss,
                perceptual_loss=perceptual_loss,
                adv_loss=adv_loss,
                scaler_g=scaler_g,
                device=device,
                kl_weight=kl_weight,
                perceptual_weight=perceptual_weight,
                adv_weight=adv_weight,
                warm_up_epochs=warm_up_epochs,
            )

            if is_main_process:
                mlflow.log_metrics({
                    "train_recon_loss": train_metrics["recon_loss"],
                    "train_kl_loss": train_metrics["kl_loss"],
                    "train_gen_loss": train_metrics["gen_loss"],
                    "train_disc_loss": train_metrics["disc_loss"],
                }, step=epoch)

            all_train_recon.append(train_metrics["recon_loss"])
            all_train_gen.append(train_metrics["gen_loss"])
            all_train_disc.append(train_metrics["disc_loss"])

            if (epoch + 1) % val_interval == 0:

                val_metrics=validate(
                    epoch=epoch,
                    autoencoder=autoencoder,
                    val_loader=val_loader,
                    l1_loss=l1_loss,
                    perceptual_loss=perceptual_loss,
                    device=device,
                    perceptual_weight=perceptual_weight,
                )

                if is_main_process:
                    mlflow.log_metrics({
                        "val_recon_loss": val_metrics["val_recon_loss"],
                        "val_p_loss": val_metrics["val_p_loss"],
                    }, step=epoch)

                print(
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"train_recon: {train_metrics['recon_loss']:.4f} | "
                    f"val_recon: {val_metrics['val_recon_loss']:.4f}"
                )

                all_val_recon.append(val_metrics["val_recon_loss"])

                is_best=val_metrics["val_recon_loss"] < best_val_loss
                if is_best:
                    best_val_loss=val_metrics["val_recon_loss"]

                save_checkpoint(
                    epoch=epoch,
                    autoencoder=autoencoder,
                    discriminator=discriminator,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    val_loss=val_metrics["val_recon_loss"],
                    save_dir=save_dir,
                    is_best=is_best,
                )

            scheduler_g.step()

            if is_main_process:
                mlflow.log_metric("lr", scheduler_g.get_last_lr()[0], step=epoch)

        total_time=time.time()-total_start
        print(f"Training completato in {total_time/3600:.2f} ore")

        if is_main_process:
            mlflow.log_metric("total_time_hours", total_time/3600)
            mlflow.log_artifact(os.path.join(save_dir, "autoencoder_best.pt"))

        curve_path=save_learning_curves(
            train_recon_loss=all_train_recon,
            val_recon_loss=all_val_recon,
            train_gen_loss=all_train_gen,
            train_disc_loss=all_train_disc,
            save_dir=save_dir,
            val_interval=val_interval,
            n_epochs=n_epochs,
        )

        if is_main_process:
            mlflow.log_artifact(curve_path)
            mlflow.end_run()

    if dist.is_initialized():
        dist.destroy_process_group()
    print("Training VAE completato")

if __name__=="__main__":
    main()