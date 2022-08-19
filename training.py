from tkinter import E
import torch 
from torch import nn
from torchvision.utils import save_image

# Get either CUDA or Mac GPU or simply CPU. 
def get_device(): 
    device = ""
    if (torch.cuda.is_available()): device = "cuda:0"
    elif (torch.has_mps): device = "mps" # This is the MacBook M1 GPU
    else: device = "cpu"
    return device

# DCGAN paper tells to initial weights from normal distribution with mean 0 and std 0.02.
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

# Train the discriminator, maximizing the probability of classifying output to be
# fake or rael. 
def train_discriminator(discriminator, optimizer, criterion, real_data, fake_data):
    batch_size = real_data.shape[0]
    real_label = torch.ones(batch_size).to(get_device())
    fake_label = torch.zeros(batch_size).to(get_device())

    optimizer.zero_grad()

    output_real = discriminator(real_data).view(-1)
    loss_real = criterion(output_real, real_label)

    output_fake = discriminator(fake_data).view(-1)
    loss_fake = criterion(output_fake, fake_label)

    loss_real.backward()

    loss_fake.backward()

    optimizer.step()

    return loss_real + loss_fake


# Train the generator, by improving its ability to generate data. 
def train_generator(discriminator, optimizer, criterion, fake_data):
    batch_size = fake_data.shape[0]
    real_label = torch.ones(batch_size).to(get_device())

    optimizer.zero_grad()

    output = discriminator(fake_data).view(-1)
    loss = criterion(output, real_label)

    loss.backward()

    optimizer.step()

    return loss

def train(generator, discriminator, optimizer_g, optimizer_d, epochs, criterion, dataloader, batch_size, input_dim):
    device = get_device()

    generator.to(device)
    generator.apply(weights_init)

    discriminator.to(device)
    discriminator.apply(weights_init)

    discriminator_loss = []
    generator_loss = []

    # Save some initial noise to see the images after each epoch
    initial_noise = torch.randn(batch_size, input_dim, 1, 1, device=device)
    print("### Beginning training ###")

    # For each epoch
    for epoch in range(epochs):
        print(f'Epoch {epoch} of {epochs}')
        # For each batch in the dataloader
        run_loss_d = 0
        run_loss_g = 0
        for i, image in enumerate(dataloader, 0):
            image = image[0].to(device)

            noise = torch.randn(batch_size, input_dim, 1, 1, device=device)
            fake_data = generator(noise)
            real_data = image

            loss_discriminator = train_discriminator(discriminator, optimizer_d, criterion, real_data, fake_data.detach()).item()
            run_loss_d += loss_discriminator

            loss_generator = train_generator(discriminator, optimizer_g, criterion, fake_data).item()
            run_loss_g += loss_generator
            
            if (i % 25 == 0):
                 print(f'- Iteration {i}: Generator loss {loss_generator} -- Discriminator loss {loss_discriminator} -')

        epoch_loss_d = run_loss_d/len(dataloader)
        epoch_loss_g = run_loss_g/len(dataloader)

        discriminator_loss.append(epoch_loss_d)
        generator_loss.append(epoch_loss_g)

        print(f'--- Epoch {epoch} avg: Generator loss {epoch_loss_g} -- Discriminator loss {epoch_loss_d} ---')

        # I save the initial noise image in each epoch
        generated_img_epoch = generator(initial_noise).cpu().detach()
        save_image(tensor=generated_img_epoch, fp=f'./saved_images/img_gen{epoch}.png', normalize=True) 

    print("### Ending training ###")
    return discriminator_loss, generator_loss