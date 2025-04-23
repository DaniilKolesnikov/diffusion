import torch

class SpatialVAE(torch.nn.Module):
    def __init__(self, latent_channels=3, inside_channels=16):
        super(SpatialVAE, self).__init__()
        self.latent_channels = latent_channels
        
        # Encoder: Downsample from 128x128 to 32x32
        self.encoder = torch.nn.Sequential(
            # Input: (3, 128, 128)
            torch.nn.Conv2d(3, inside_channels, kernel_size=4, stride=2, padding=1),   # -> (32, 64, 64)
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(inside_channels),

            torch.nn.Conv2d(inside_channels, inside_channels * 2, kernel_size=4, stride=2, padding=1),  # -> (64, 32, 32)
            torch.nn.BatchNorm2d(inside_channels * 2),
            torch.nn.ReLU(inplace=True)
        )
        # Two convolutional heads to produce the mean (mu) and log-variance (logvar)
        # Both outputs have shape: (latent_channels, 32, 32)
        self.fc_mu = torch.nn.Conv2d(inside_channels * 2, latent_channels, kernel_size=3, stride=1, padding=1)
        self.fc_logvar = torch.nn.Conv2d(inside_channels * 2, latent_channels, kernel_size=3, stride=1, padding=1)
        
        # Decoder: Upsample from latent (3, 32, 32) back to image (3, 128, 128)
        self.decoder = torch.nn.Sequential(
            # Input: (latent_channels, 32, 32)
            torch.nn.ConvTranspose2d(latent_channels, inside_channels * 2, kernel_size=4, stride=2, padding=1),  # -> (64, 64, 64)
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(inside_channels * 2),

            torch.nn.ConvTranspose2d(inside_channels * 2, inside_channels, kernel_size=4, stride=2, padding=1),               # -> (32, 128, 128)
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(inside_channels),

            torch.nn.Conv2d(inside_channels, 3, kernel_size=3, stride=1, padding=1),                          # -> (3, 128, 128)
            torch.nn.Tanh()  # Ensure output pixel values are in [-1,1]
        )

    def encode(self, x):
        enc = self.encoder(x)
        mu = self.fc_mu(enc)
        logvar = self.fc_logvar(enc)
        return mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        # Compute standard deviation and sample epsilon (element-wise)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        batch_size = x.size(0)
        # Encode input image to feature maps of shape (batch, 128, 32, 32)
        enc = self.encoder(x)
        # Compute per-pixel (and per-channel) mu and logvar: (batch, 3, 32, 32)
        mu = self.fc_mu(enc)
        logvar = self.fc_logvar(enc)
        # Reparameterize: sample z ~ N(mu, exp(logvar))
        z = self.reparameterize(mu, logvar)
        # Decode latent tensor back to image shape: (batch, 3, 128, 128)
        recon = self.decoder(z)
        return recon, mu, logvar
    
def train_vae(model, loader, num_epochs=10, learning_rate=0.001):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    # Initialize the GradScaler for mixed precision training
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, _ in tqdm(loader, total=len(loader)):
            images = images.to(device)
            optimizer.zero_grad()
            
            # Use autocast for mixed precision
            with autocast():
                recon, mu, logvar = model(images)
                loss = criterion(recon, images)
                # Compute KL divergence between the latent distribution and a standard normal distribution
                kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss += 0.001 * kl_div
            
            # Scale the loss and call backward
            scaler.scale(loss).backward()
            
            # Update weights with the scaled gradients
            scaler.step(optimizer)
            
            # Update the scaler for next iteration
            scaler.update()
            
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(loader)}')
    print('Finished Training')