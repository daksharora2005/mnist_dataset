
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
    print(f"DCGAN Epoch {epoch+1}/{epochs} | D: {loss_d.item():.4f} | G: {loss_g.item():.4f}")

G3.eval()
G4.eval()
with torch.no_grad():
    vanilla_samples = G3(torch.randn(8, 64)).view(-1, 1, 28, 28).numpy()
    dc_samples = G4(torch.randn(8, 64)).numpy()

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i in range(8):
    axes[0, i].imshow(vanilla_samples[i, 0], cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(dc_samples[i, 0], cmap='gray')
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
