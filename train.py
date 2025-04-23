import torch
from tqdm import tqdm
from utils import get_clip_text_embedding
from torch.cuda.amp import autocast, GradScaler

MAX_TIMESTEPS = 100

def train_one_epoch_unconditional(unet, vae, optimizer, loader, noise_scheduler, device):
    unet.train()
    losses = []
    for images, _ in tqdm(loader, total=len(loader)):

        # images = augment_images(images).to(device)
        images = vae.reparameterize(*vae.encode(images.to(device))).to(device)

        img_shape = images.shape

        timesteps = torch.randint(0, MAX_TIMESTEPS, (img_shape[0],)).to(device)
        noise = torch.randn(img_shape).to(device)

        noisy_images = noise_scheduler.add_noise(images, noise, timesteps).to(device)

        prediction = unet(noisy_images, timesteps, return_dict=False)[0]

        loss = torch.nn.functional.l1_loss(prediction, noise)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

    return losses


def estimate_loss(unet, vae, dataloader, device, noise_scheduler, clip_tokenizer, clip_text_encoder, num_samples=50):
    """
    Estimate the loss for a batch of images and their corresponding text embeddings.
    This function is used to compute the loss for the UNet model during training.

    Args:
        unet (torch.nn.Module): The UNet model.
        vae (torch.nn.Module): The VAE model.
        num_samples (int): Number of samples to estimate the loss.

    Returns:
        torch.Tensor: The estimated loss.
    """
    unet.eval()
    vae.eval()
    losses = torch.zeros(num_samples).to(device)

    scaler = GradScaler()

    with torch.no_grad():
        for i in range(num_samples):
            # Sample a batch of images and their corresponding text embeddings
            images, attr_texts = next(iter(dataloader))
            images = images.to(device)
            text_embeddings = get_clip_text_embedding(attr_texts, clip_tokenizer, clip_text_encoder, device).to(device)

            # Encode the images using the VAE
            mu, logvar = vae.encode(images)
            z = vae.reparameterize(mu, logvar)

            # Sample random timesteps
            timesteps = torch.randint(0, MAX_TIMESTEPS, (images.size(0),)).to(device)
            noise = torch.randn_like(z)

            # Add noise to the images
            noisy_images = noise_scheduler.add_noise(z, noise, timesteps).to(device)

            with autocast():
                # Get the model output
                model_output = unet(noisy_images, timesteps, encoder_hidden_states=text_embeddings).sample
                # Compute the loss
                loss = torch.nn.functional.l1_loss(model_output, noise)
                losses[i] = loss
            
    
    unet.train()
    vae.train()
    return losses.mean()

def train_one_epoch_with_text(unet, vae, optimizer, loader, device, noise_scheduler, all_losses, clip_tokenizer, clip_text_encoder, scheduler=None):
    """
    Train the diffusion UNet for one epoch with text conditioning.
    The images are first encoded via the VAE, noise is added,
    and then the UNet predicts the noise conditioned on CLIP text embeddings.
    
    Args:
        unet (torch.nn.Module): The diffusion model.
        vae (torch.nn.Module): The VAE that encodes images into latents.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loader (DataLoader): DataLoader yielding (image, attr_text) pairs.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
    
    Returns:
        list: A list of loss values for the epoch.
    """
    unet.train()
    losses = []
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    for step, (images, attr_text) in enumerate(tqdm(loader, total=len(loader))):
        images = images.to(device)
        
        # Get CLIP text embeddings for conditioning.
        text_embeddings = get_clip_text_embedding(attr_text, clip_tokenizer, clip_text_encoder, device).to(device)
        
        # Encode images to latent space via the VAE.
        with torch.no_grad():
            latents = vae.reparameterize(*vae.encode(images)).to(device)
        
        img_shape = latents.shape
        
        # Sample random timesteps and noise.
        timesteps = torch.randint(0, MAX_TIMESTEPS, (img_shape[0],), device=device)
        noise = torch.randn(img_shape, device=device)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        optimizer.zero_grad()
        
        # Use autocast for mixed precision forward pass
        with autocast():
            # Forward pass through UNet with text conditioning
            prediction = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings, return_dict=False)[0]
            loss = torch.nn.functional.l1_loss(prediction, noise)
        
        # Scale gradients and perform backward pass
        scaler.scale(loss).backward()
        
        # Update weights with scaled gradients
        scaler.step(optimizer)
        
        # Update scaler for next iteration
        scaler.update()
        
        # Step the scheduler if provided
        if scheduler is not None:
            scheduler.step()
        
        losses.append(loss.item())
        all_losses.append(loss.item())


        if step % 50 == 0:
            estimated_loss = estimate_loss(unet, vae, loader)
            if estimated_loss < best_loss:
                best_loss = estimated_loss
                torch.save(unet.state_dict(), "best_unet_cond.pth")
                print(f"New best model saved with loss: {best_loss:.4f}")

            print(f"Step {step}/{len(loader)}, Loss: {estimated_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        if step % 100 == 0:
            # Save intermediate model state
            torch.save(unet.state_dict(), f"checkpoints/unet_cond_step_{step}.pth")
    
    return losses