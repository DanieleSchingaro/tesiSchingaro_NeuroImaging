#tests/test_vae.py
"""
Test rapido del VAE.
Esegui con: python3 -m tests.test_vae
"""
import torch
from torch.amp import GradScaler, autocast
from monai.utils import set_determinism
from src.training.train_vae import load_config, setup_models, setup_losses
from src.data.dataset import get_file_list, DATASET_DIRS, create_dataloader

def test_vae(n_batches: int = 2):
    print("Test rapido VAE...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(seed=42)

    files = get_file_list(DATASET_DIRS)[:10]
    train_loader = create_dataloader(
        files[:8],
        patch_size=(64, 64, 64),
        batch_size=1,
        is_train=True,
        random_aug=False,
        cache_rate=0.0,
        num_workers=2,
    )

    config_vae = load_config("configs/config_vae.json")
    autoencoder, discriminator = setup_models(config_vae, device)
    l1_loss, perceptual_loss, adv_loss = setup_losses(device)

    autoencoder.train()
    for i, batch in enumerate(train_loader):
        if i >= n_batches:
            break
        images = batch["image"].to(device)
        with autocast("cuda", enabled=True):
            reconstruction, z_mu, z_sigma = autoencoder(images)
            recon_loss = l1_loss(reconstruction.float(), images.float())
        print(f"Batch {i+1}: input={images.shape}, "
              f"recon={reconstruction.shape}, "
              f"latent_mu={z_mu.shape}, "
              f"loss={recon_loss.item():.4f}")
        assert reconstruction.shape == images.shape, "Shape ricostruzione non corretta!"
        assert z_mu.shape == (1, 4, 16, 16, 16), "Shape latente non corretta!"

    print("Test VAE superato!")

if __name__ == "__main__":
    test_vae()
