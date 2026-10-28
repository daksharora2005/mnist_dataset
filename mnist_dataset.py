
# //1+2: Set up PyTorch, run a basic GAN example, implement simple generator and discriminator, verify forward pass with random input

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 784), nn.Tanh())
    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x):
        return self.net(x)

G = Generator()
D = Discriminator()
z = torch.randn(4, 64)
fake = G(z)
pred = D(fake)
print("Generator output shape:", fake.shape)
print("Discriminator output shape:", pred.shape)
print("Discriminator predictions:", pred.detach().numpy().flatten())


# //3+8: Train a vanilla GAN on MNIST, track losses, observe generated samples evolution, and evaluate using FID and Inception Score

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(mnist, batch_size=128, shuffle=True)

z_dim = 64

class VanillaG(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 784), nn.Tanh())
    def forward(self, x):
        return self.net(x)

class VanillaD(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(256, 1), nn.Sigmoid())
    def forward(self, x):
        return self.net(x)

G3 = VanillaG()
D3 = VanillaD()
opt_g = optim.Adam(G3.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = optim.Adam(D3.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

g_losses, d_losses = [], []
sample_images = []
epochs = 10

for epoch in range(epochs):
    for real, _ in loader:
        bs = real.size(0)
        real_flat = real.view(bs, -1)
        ones = torch.full((bs, 1), 0.9)
        zeros = torch.zeros(bs, 1)

        opt_d.zero_grad()
        d_real = D3(real_flat)
        z = torch.randn(bs, z_dim)
        fake = G3(z)
        d_fake = D3(fake.detach())
        loss_d = criterion(d_real, ones) + criterion(d_fake, zeros)
        loss_d.backward()
        opt_d.step()

        opt_g.zero_grad()
        d_fake2 = D3(fake)
        loss_g = criterion(d_fake2, torch.ones(bs, 1))
        loss_g.backward()
        opt_g.step()

    g_losses.append(loss_g.item())
    d_losses.append(loss_d.item())
    G3.eval()
    with torch.no_grad():
        sample = G3(torch.randn(16, z_dim)).view(-1, 1, 28, 28)
        sample_images.append(sample)
    G3.train()
    print(f"Epoch {epoch+1}/{epochs} | D Loss: {loss_d.item():.4f} | G Loss: {loss_g.item():.4f}")

plt.figure()
plt.plot(g_losses, label='G Loss')
plt.plot(d_losses, label='D Loss')
plt.legend()
plt.title('Vanilla GAN Losses')
plt.show()

fig, axes = plt.subplots(epochs, 1, figsize=(8,15))
for i, imgs in enumerate(sample_images):
    row = np.concatenate(imgs[:8].squeeze().numpy(), axis=1)
    axes[i].imshow(row, cmap='gray')
    axes[i].set_title(f'Epoch {i+1}')
    axes[i].axis('off')
plt.suptitle('Generated Samples Evolution')
plt.show()

def prep_for_metrics(imgs):
    imgs = imgs.repeat(1, 3, 1, 1)
    imgs = torch.nn.functional.interpolate(imgs, size=(299, 299))
    return ((imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)

G3.eval()
with torch.no_grad():
    real_batch = torch.stack([mnist[i][0] for i in range(100)])
    fake_batch = G3(torch.randn(100, z_dim)).view(-1, 1, 28, 28)

fid = FrechetInceptionDistance(feature=64)
fid.update(prep_for_metrics(real_batch), real=True)
fid.update(prep_for_metrics(fake_batch), real=False)
print(f"FID Score: {fid.compute().item():.2f}")

isc = InceptionScore()
isc.update(prep_for_metrics(fake_batch))
is_mean, is_std = isc.compute()
print(f"Inception Score: {is_mean:.4f} +/- {is_std:.4f}")


# //4: Build and train a DCGAN for image generation, compare with basic GAN

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(mnist, batch_size=128, shuffle=True)

z_dim = 64

class DCGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256*4*4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh()
        )
    def forward(self, x):
        x = self.fc(x).view(-1, 256, 4, 4)
        return self.net(x)

class DCDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(256*4*4, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

G4 = DCGenerator()
D4 = DCDiscriminator()
opt_g = optim.Adam(G4.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = optim.Adam(D4.parameters(), lr=2e-4, betas=(0.5, 0.999))
G_basic = nn.Sequential(nn.Linear(z_dim, 256), nn.ReLU(), nn.Linear(256, 1024), nn.Tanh())
D_basic = nn.Sequential(nn.Linear(1024, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1), nn.Sigmoid())
opt_gb = optim.Adam(G_basic.parameters(), lr=2e-4)
opt_db = optim.Adam(D_basic.parameters(), lr=2e-4)
criterion = nn.BCELoss()
epochs = 5

for epoch in range(epochs):
    for real, _ in loader:
        bs = real.size(0)
        ones = torch.ones(bs, 1)
        zeros = torch.zeros(bs, 1)

        opt_d.zero_grad()
        d_real = D4(real)
        z = torch.randn(bs, z_dim)
        fake = G4(z)
        d_fake = D4(fake.detach())
        loss_d = criterion(d_real, ones) + criterion(d_fake, zeros)
        loss_d.backward()
        opt_d.step()

        opt_g.zero_grad()
        loss_g = criterion(D4(fake), ones)
        loss_g.backward()
        opt_g.step()

        real_flat = real.view(bs, -1)
        opt_db.zero_grad()
        fake_b = G_basic(z)
        loss_db = criterion(D_basic(real_flat), ones) + criterion(D_basic(fake_b.detach()), zeros)
        loss_db.backward()
        opt_db.step()

        opt_gb.zero_grad()
        loss_gb = criterion(D_basic(fake_b), ones)
        loss_gb.backward()
        opt_gb.step()
    print(f"DCGAN Epoch {epoch+1}/{epochs} | D: {loss_d.item():.4f} | G: {loss_g.item():.4f}")

G_basic.eval()
G4.eval()
with torch.no_grad():
    vanilla_samples = G_basic(torch.randn(8, z_dim)).view(-1, 32, 32).numpy()
    dc_samples = G4(torch.randn(8, z_dim)).squeeze().numpy()

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i in range(8):
    axes[0, i].imshow(vanilla_samples[i], cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(dc_samples[i], cmap='gray')
    axes[1, i].axis('off')
axes[0, 0].set_ylabel('Vanilla')
axes[1, 0].set_ylabel('DCGAN')
plt.suptitle('Vanilla GAN vs DCGAN Comparison')
plt.show()


# //5: Implement a Conditional GAN (CGAN) to generate labeled digits and test performance

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(mnist, batch_size=128, shuffle=True)

z_dim, n_classes, emb_dim = 64, 10, 32

class CGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, emb_dim)
        self.net = nn.Sequential(nn.Linear(z_dim + emb_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 784), nn.Tanh())
    def forward(self, z, labels):
        x = torch.cat([z, self.label_emb(labels)], dim=1)
        return self.net(x)

class CDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, emb_dim)
        self.net = nn.Sequential(nn.Linear(784 + emb_dim, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(256, 1), nn.Sigmoid())
    def forward(self, x, labels):
        x = torch.cat([x, self.label_emb(labels)], dim=1)
        return self.net(x)

G5 = CGenerator()
D5 = CDiscriminator()
opt_g = optim.Adam(G5.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = optim.Adam(D5.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()
epochs = 5

for epoch in range(epochs):
    for real, labels in loader:
        bs = real.size(0)
        real_flat = real.view(bs, -1)
        ones = torch.full((bs, 1), 0.9)
        zeros = torch.zeros(bs, 1)

        opt_d.zero_grad()
        z = torch.randn(bs, z_dim)
        fake = G5(z, labels)
        loss_d = criterion(D5(real_flat, labels), ones) + criterion(D5(fake.detach(), labels), zeros)
        loss_d.backward()
        opt_d.step()

        opt_g.zero_grad()
        loss_g = criterion(D5(fake, labels), torch.ones(bs, 1))
        loss_g.backward()
        opt_g.step()
    print(f"CGAN Epoch {epoch+1}/{epochs} | D: {loss_d.item():.4f} | G: {loss_g.item():.4f}")

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
G5.eval()
with torch.no_grad():
    imgs = G5(torch.randn(10, z_dim), torch.arange(10)).view(10, 28, 28).numpy()
    for digit in range(10):
        ax = axes[digit // 5, digit % 5]
        ax.imshow(imgs[digit], cmap='gray')
        ax.set_title(f'{digit}')
        ax.axis('off')
plt.suptitle('CGAN - Conditional Generation (digits 0-9)')
plt.show()


# //7 style gan
import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MiniStyleGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mapping = nn.Sequential(nn.Linear(32, 32), nn.LeakyReLU(0.2), nn.Linear(32, 32))
        self.const = nn.Parameter(torch.randn(1, 32, 4, 4))
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.mod1 = nn.Linear(32, 32*2)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear')
        self.to_img = nn.Sequential(nn.Conv2d(32, 1, 3, padding=1), nn.Tanh())

    def forward_w(self, w):
        b = w.size(0)
        x = self.conv1(self.const.repeat(b, 1, 1, 1))
        # AdaIN
        s = self.mod1(w).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = s[:, :32], s[:, 32:]
        x = (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-5) * (gamma +
1.0) + beta
        return self.to_img(self.up(x))

model = MiniStyleGAN().to(device)

with torch.no_grad():

    w1 = model.mapping(torch.randn(1, 32, device=device))
    w2 = model.mapping(torch.randn(1, 32, device=device))
    steps = 8
    alphas = torch.linspace(0, 1, steps, device=device).unsqueeze(1)
    w_interp = (1 - alphas) * w1 + alphas * w2
    images = model.forward_w(w_interp).cpu()

fig, axes = plt.subplots(1, steps, figsize=(14, 2))
for i, ax in enumerate(axes):
    ax.imshow(images[i].squeeze(), cmap='gray')
    ax.set_title(f"t={i/(steps-1):.2f}")
    ax.axis('off')
plt.suptitle("StyleGAN Latent Interpolation Strip")
plt.show()
print("StyleGAN completed successfully!")


# cycle gan
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1), nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

G_AB, G_BA = G().to(device), G().to(device)
D_A = nn.Sequential(nn.Conv2d(1, 8, 4, 2, 1), nn.LeakyReLU(0.2),
                    nn.Flatten(), nn.Linear(8*14*14, 1), nn.Sigmoid()).to(device)
D_B = nn.Sequential(nn.Conv2d(1, 8, 4, 2, 1), nn.LeakyReLU(0.2),
                    nn.Flatten(), nn.Linear(8*14*14, 1), nn.Sigmoid()).to(device)

opt_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=0.001)
opt_D = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=0.001)
bce, l1 = nn.BCELoss(), nn.L1Loss()

# Create two simple image domains
A = torch.zeros(16, 1, 28, 28, device=device)
A[:, :, 7:21, 7:21] = 1
B = 1 - A


for epoch in range(50):
    fake_B = G_AB(A)
    fake_A = G_BA(B)
    rec_A, rec_B = G_BA(fake_B), G_AB(fake_A)

    opt_G.zero_grad()
    loss_G = (bce(D_B(fake_B), torch.ones(16, 1, device=device)) +
              bce(D_A(fake_A), torch.ones(16, 1, device=device)) +
              10 * (l1(rec_A, A) + l1(rec_B, B)))
    loss_G.backward()
    opt_G.step()

    opt_D.zero_grad()
    loss_D = (bce(D_A(A), torch.ones(16, 1, device=device)) +
              bce(D_A(fake_A.detach()), torch.zeros(16, 1, device=device)) +
              bce(D_B(B), torch.ones(16, 1, device=device)) +
              bce(D_B(fake_B.detach()), torch.zeros(16, 1, device=device)))
    loss_D.backward()
    opt_D.step()

with torch.no_grad():
    translated = G_AB(A[:1])
    reconstructed = G_BA(translated)

plt.figure(figsize=(8, 3))
for i, img in enumerate([A[0], translated[0], reconstructed[0]]):
    plt.subplot(1, 3, i+1)
    plt.imshow(img.squeeze().cpu(), cmap="gray")
    plt.axis("off")
plt.suptitle("Source A | Translated B | Reconstructed A")
plt.show()

print("Generator loss:", round(loss_G.item(), 4))
print("Discriminator loss:", round(loss_D.item(), 4))


# super resolution dataset
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. SRGAN Model
class SRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.PReLU(),
            nn.Conv2d(32, 32*4, 3, padding=1), nn.PixelShuffle(2), nn.PReLU(),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x): return self.net(x)

model = SRNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)

# 2. Data (14x14 Low-Res -> 28x28 High-Res)
hr_data = torch.randn(200, 1, 28, 28).clamp(0, 1).to(device)
lr_data = F.interpolate(hr_data, size=(14, 14), mode='bicubic').clamp(0, 1)

# 3. Fast Train (40 steps)
for step in range(40):
    opt.zero_grad()
    loss = F.mse_loss(model(lr_data), hr_data)
    loss.backward(); opt.step()

# 4. Evaluate PSNR & Visuals
with torch.no_grad():
    sr_out = model(lr_data[:1]).cpu().numpy()
    bicubic = F.interpolate(lr_data[:1], size=(28, 28), mode='bicubic').cpu().numpy()
    real_hr = hr_data[:1].cpu().numpy()

def psnr(a, b): return 20 * np.log10(1.0 / np.sqrt(np.mean((a - b)**2)))

fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))
axes[0].imshow(lr_data[0].squeeze().cpu(), cmap='gray'); axes[0].set_title("Low-Res (14x14)");
axes[0].axis('off')
axes[1].imshow(bicubic[0].squeeze(), cmap='gray'); axes[1].set_title(f"Bicubic ({psnr(bicubic,
real_hr):.1f}dB)"); axes[1].axis('off')
axes[2].imshow(sr_out[0].squeeze(), cmap='gray'); axes[2].set_title(f"SRGAN ({psnr(sr_out,
real_hr):.1f}dB)"); axes[2].axis('off')
axes[3].imshow(real_hr[0].squeeze(), cmap='gray'); axes[3].set_title("Target HR");
axes[3].axis('off')
plt.suptitle("Super-Resolution Fast Pipeline")
plt.show()
print("SRGAN completed successfully!")
