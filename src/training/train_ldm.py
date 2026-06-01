#src/training/train_ldm.py
"""
Training dell'LDM sui latenti generati dal VAE.
Basato su diff_model_train.py di NV-Generate-CTMR (MAISI)

Obiettivo: generare MRI cerebrali T1 skull-stripped sintetiche di soggetti sani (HC).
Il modello è INCONDIZIONATO:
    -niente body region (top/bottom region index)
    -niente modality embedding
    -niente spacing input
Genera HC dal rumore puro.

Scheduler: RFlowScheduler (rectified flow), target=images-noise
scale_factor calcolato sul primo batch (1/std(z)) e salvato nel checkpoint,
serve anche in sampling per de-normalizzare (approcio MAISI)
"""

import os
import json
import argparse
import torch
import torch.distributed as dist 
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from monai.utils import set_determinism, first
from monai.networks.schedulers import RFlowScheduler
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUnetMaisi 
from tqdm import tqdm
import mlflow
from src.data.ldm_dataset import setup_ldm_dataloaders

#DDP setup
def setup_ddp()->tuple[int, torch.device]:
    """
    Inizializza DDP con torchrun (LOCAL_RANK, RANK, WORLD_SIZE già settati)
    """
    dist.init_process_group(backend="nccl")
    local_rank=int(os.environ(["LOCAL_RANK"]))
    torch.cuda.set_device(local_rank)
    device=torch.device(f"cuda:{local_rank}")
    return local_rank, device

#Modello UNet di diffusione
def setup_unet(net_cfg:dict, device:torch.device, local_rank:int)->torch.nn.Module:
    """
    Inizializza la Unet di diffusione incondizionata.
    Tutti i conditioning sono DISATTIVATI per la generazione HC pura:
        - include_top/bottom_region_index_input=False  (niente body region)
        - include_spacing_input=False                  (niente spacing)
        - num_class_embeds=None                        (niente modality)
        - with_conditioning=False                      (niente cross-attention)
    Con questi flag il forward si chiama con solo (x, timesteps).
 
    Parametri letti da config_network["diffusion_unet_def"]. Le chiavi "_target_"
    e i riferimenti "@..." del config bundle vengono ignorati: passiamo i valori
    espliciti.
    """
    unet=DiffusionModelUnetMaisi(
        spatial_dims=net_cfg.get("spatial_dims", 3),
        in_channels=net_cfg.get("in_channels", 4),
        out_channels=net_cfg.get("out_channels", 4),
        num_channels=net_cfg.get("num_channels", [64, 128, 256, 512]),
        attention_levels=net_cfg.get("attention_levels", [False, False, True, True]),
        num_head_channels=net_cfg.get("num_head_channels", [0, 0, 32, 32]),
        num_res_blocks=net_cfg.get("num_res_blocks", 2),
        use_flash_attention=net_cfg.get("use_flash_attention", True),
        resblock_updown=net_cfg.get("resblock_updown", True),
        include_fc=net_cfg.get("include_fc", True),
        # conditioning disattivati (generazione HC incondizionata)
        with_conditioning=False,
        num_class_embeds=None,
        include_top_region_index_input=False,
        include_bottom_region_index_input=False,
        include_spacing_input=False,
    ).to(device)

    if dist.is_available() and dist.is_initialized():
        unet=DistributedDataParallel(
            unet, device_ids=[local_rank], find_unused_parameters=False,
        )
    return unet

def setup_noise_scheduler(sched_cfg: dict)->RFlowScheduler:
    """
    Crea il RFlowScheduler con i parametri del config (MAISI-style):
    num_train_timesteps, use_discrete_timesteps, use_timesteps_transform, scale, sample_method.
    """
    return RFlowScheduler(
        num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000),
        use_discrete_timesteps=sched_cfg.get("use_discrete_timesteps", False),
        use_timestep_transform=sched_cfg.get("use_timestep_transform", True),
        scale=sched_cfg.get("scale", 1.4),
        sample_method=sched_cfg.get("sample_method", "uniform"),
    )

#scale_factor (MAISI-style)
def calculate_scale_factor(train_loader, device:torch.device)->torch.Tensor:
    """
    scale_factor=1/std(z) calcolato sul primo batch.
    Su DDP viene mediato tra i rank con all_reduce AVG.
    Serve a normalizzare i latenti (varianza circa 1) per la diffusione,
    e va salvato nel checkpoint per de-normalizzare in sampling.
    """
    check_data=first(train_loader)
    z=check_data["latent"].to(device)
    scale_factor=1.0/torch.std(z)

    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)

    return scale_factor

def train_one_epoch(
    epoch,unet,train_loader,optimizer,lr_scheduler,
    loss_pt, scaler, scale_factor, noise_scheduler,
    device, local_rank, amp=True,
):
    """
    Singola epoca di training rectified flow.
    target=images-noise (velocity lungo il patch lineare)
    """
    unet.train()
    loss_acc=torch.zeros(2, dtype=torch.float, device=device)

    #barra di avanzamento solo su rank 0
    progress_bar=tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}",
        ncols=100,
        disable=(local_rank!=0),
    )

    for train_data in progress_bar:
        #latente grezzo -> normalizzato con scale_factor
        images=train_data["latent"].to(device)
        images=images*scale_factor

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=amp):
            noise=torch.randn_like(images)

            #RFlow: timesteps campionati dallo scheduler
            timesteps=noise_scheduler.sample_timesteps(images)

            #aggiunge rumore lungo il path lineare
            noisy_latent=noise_scheduler.add_noise(
                original_samples=images, noise=noise, timesteps=timesteps,
            )

            #Unet incondizionata: solo x e timesteps
            model_output=unet(x=noisy_latent, timesteps=timesteps)

            #target rectified flow (velocity)
            model_gt=images-noise

            loss=loss_pt(model_output.float(), model_gt.float())
        
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        lr_scheduler.step()
        
        loss_acc[0]+=loss.item()
        loss_acc[1]+=1.0

        #aggiornamento della barra con la loss media corrente
        if local_rank==0:
            progress_bar.set_postfix({"loss": f"{(loss_acc[0]/loss_acc[1]).item():.5f}"})
    
    if dist.is_initialized():
        dist.all_reduce(loss_acc, op=torch.distributed.ReduceOp.SUM)
    
    return (loss_acc[0]/loss_acc[1]).item()

#Validazione
@torch.no_grad() #disattiva il calcolo dei gradienti
def validate(unet, val_loader, loss_pt, scale_factor, noise_scheduler, device, amp=True):
    """
    Loss di validazione (stesso obiettivo di rectified flow, senza backward)
    NB: per i modelli di diffusione la val_loss e' un indicatore debole della
    qualita' di generazione. La valutazione vera (FID, MMD, MS-SSIM) si fa sui
    campioni generati in eval.py. Qui serve solo a monitorare l'overfitting.
    """
    unet.eval()
    loss_acc=torch.zeros(2, dtype=torch.float, device=device)

    for val_data in val_loader:
        images=val_data["latent"].to(device)
        images=images*scale_factor

        with autocast("cuda", enabled=amp):
            noise=torch.randn_like(images)
            timesteps=noise_scheduler.sample_timesteps(images)
            noisy_latent=noise_scheduler.add_noise(
                original_samples=images, noise=noise, timesteps=timesteps,
            )
            model_output=unet(x=noisy_latent, timesteps=timesteps)
            model_gt=images-noise
            loss=loss_pt(model_output.float(), model_gt.float())

        loss_acc[0]+=loss.item()
        loss_acc[1]+=1.0
    
    if dist.is_initialized():
        dist.all_reduce(loss_acc, op=torch.distributed.ReduceOp.SUM)
    
    return (loss_acc[0]/loss_acc[1]).item()

#Checkpoint
def save_checkpoint(epoch, unet, loss, scale_factor, num_train_timesteps, save_path):
    """
    Salva il checkpoint. scale_factor e num_train_timesteps inclusi: servono
    entrambi al sampling per ricostruire lo scheduler e de-normalizzare.
    """
    unet_state=unet.module.state_dict() if dist.is_initialized() else unet.state_dict()
    torch.save(
        {
            "epoch":epoch+1,
            "loss":loss,
            "num_train_timesteps":num_train_timesteps,
            "scale_factor":scale_factor,
            "unet_state_dict":unet_state,
        },
        save_path,
    )

def main():
    parser=argparse.ArgumentParser(description="Training LDM su latenti del VAE")
    parser.add_argument("--config", type=str, default="configs/config_diff_model.json")
    parser.add_argument("--network", type=str, default="configs/config_network.json")
    args=parser.parse_args()

    #caricamento delle config
    with open(args.config) as f:
        config=json.load(f) #diffusion_unet_train, paths ...
    with open(args.network) as f:
        config_net=json.load(f) #diffusion_unet_def, noise_scheduler ...
    
    #DDP
    local_rank, device=setup_ddp()
    is_main=local_rank==0
    set_determinism(seed=42)

    if is_main:
        print(f"Device: {device}, GPU: {torch.cuda.device_count()}")
        mlflow.set_experiment("LDM_training")
    
    #parametri di training (config_diff_model["diffusion_unet_train"])
    train_cfg=config["diffusion_unet_train"]
    n_epochs=train_cfg.get("n_epochs", 1000)
    lr=train_cfg.get("lr", 1e-5)
    batch_size=train_cfg.get("batch_size", 1)
    num_workers=train_cfg.get("num_workers", 4)
    val_interval=train_cfg.get("val_interval", 50)     # default 50 se assente
    save_interval=train_cfg.get("save_interval", 100)  # checkpoint periodici
    amp=train_cfg.get("amp", True)

     #path (config_diff_model["paths"])
    paths=config["paths"]
    save_dir=paths.get("model_dir", "./outputs/models")
    splits_path=paths["splits_path"]   # embeddings_dataset.json
    os.makedirs(save_dir, exist_ok=True)
 
    #scheduler params (config_network["noise_scheduler"])
    sched_cfg=config_net["noise_scheduler"]
    num_train_timesteps=sched_cfg.get("num_train_timesteps", 1000)

    #dataloaders
    loader_cfg={
        "batch_size":batch_size,
        "num_workers":num_workers,
        "divisible_k":8,
    }
    train_loader, val_loader, _=setup_ldm_dataloaders(loader_cfg, splits_path)

    #Unet (config_network["diffusion_unet_def"])
    unet=setup_unet(config_net["diffusion_unet_def"], device, local_rank)

    #scale_factor (MAISI) su primo batch
    scale_factor=calculate_scale_factor(train_loader, device)
    if is_main:
        print(f"scale_factor={scale_factor.item():.5f}")
    
    #scheduler RFlow
    noise_scheduler=setup_noise_scheduler(sched_cfg)

    #optimizer+lr scheduler (MAISI: Adam+PolynomialLR power 2.0)
    optimizer=torch.optim.Adam(params=unet.parameters(), lr=lr)
    total_steps=n_epochs*len(train_loader)
    lr_scheduler=torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=total_steps, power=2.0,
    )

    loss_pt=torch.nn.L1Loss()
    scaler=GradScaler("cuda", enabled=amp)

    best_val_loss=float("inf")

    #loop di training
    for epoch in range(n_epochs):
        train_loss=train_one_epoch(
            epoch, unet, train_loader, optimizer, lr_scheduler,
            loss_pt, scaler, scale_factor, noise_scheduler,
            device, local_rank, amp=amp,
        )

        if is_main:
            current_lr=optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{n_epochs} | train_loss: {train_loss:.5f} | lr: {current_lr:.2e}")
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("lr", current_lr, step=epoch)
            #salva sempre l'ultimo
            save_checkpoint(
                epoch, unet, train_loss, scale_factor, num_train_timesteps,
                os.path.join(save_dir, "ldm_unet_last.pt"),
            )

            #checkpoint periodico (per scegliere poi il best con FID, non con val_loss)
            if (epoch+1)%save_interval==0:
                save_checkpoint(
                    epoch, unet, train_loss, scale_factor, num_train_timesteps,
                    os.path.join(save_dir, f"ldm_unet_epoch{epoch+1}.pt"),
                )
        
        # validazione periodica (monitoraggio overfitting)
        if (epoch+1) % val_interval==0:
            val_loss=validate(
                unet, val_loader, loss_pt, scale_factor, noise_scheduler, device, amp=amp,
            )
            if is_main:
                print(f"  -> val_loss: {val_loss:.5f}")
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                if val_loss<best_val_loss:
                    best_val_loss=val_loss
                    save_checkpoint(
                        epoch, unet, val_loss, scale_factor, num_train_timesteps,
                        os.path.join(save_dir, "ldm_unet_best.pt"),
                    )
                    print(f"  -> nuovo best (val_loss={val_loss:.5f}), salvato ldm_unet_best.pt")
 
    if dist.is_initialized():
        dist.destroy_process_group()
 
 
if __name__=="__main__":
    main()