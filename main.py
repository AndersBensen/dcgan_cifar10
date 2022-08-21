from models import Generator, Discriminator
import training 
from torch import nn, optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

if __name__ == "__main__":
    latent_dim = 100
    generator = Generator(latent_dim)
    discriminator = Discriminator()

    lr = 0.0002
    beta1 = 0.5

    optimizer_g = optim.Adam(generator.parameters(), lr, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr, betas=(beta1, 0.999))
    
    criterion = nn.BCELoss()

    epochs = 25
    batch_size = 128
    image_size = 64

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_data = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    print('Dataset size:', len(train_data))

    discriminator_loss, generator_loss = training.train(
                generator=generator, discriminator=discriminator, optimizer_d=optimizer_d, optimizer_g=optimizer_g, 
                epochs=epochs, criterion=criterion, batch_size=batch_size, dataloader=dataloader, latent_dim=latent_dim)