import os

import numpy as np
import torch
import torchvision.utils as vutils
from torch import nn
from torch.autograd import Variable
from torch.nn import Module
from torch.nn.utils import spectral_norm
from tqdm import tqdm

from networks.DiffAugment_pytorch import DiffAugment

policy = 'translation,cutout'


class latant_mapping(nn.Module):
    def __init__(self, z_dim):
        super(latant_mapping, self).__init__()
        self.z_dim = z_dim

        self.model = nn.Sequential(
            nn.Linear(self.z_dim,self.z_dim),
            nn.BatchNorm1d(self.z_dim)
            )


    def forward(self, z):
        return self.model(z)

class Generator_MNIST(Module):
    def __init__(self, nz):
        super(Generator_MNIST, self).__init__()
        self.nz = nz
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(self.nz,56*56*1)
        self.BN1 = nn.BatchNorm2d(1)


        self.deconv2 = spectral_norm(nn.Conv2d(1,2*32, 3, 1, 1))
        self.BN2 = nn.BatchNorm2d(2*32)

        self.deconv3 = spectral_norm(nn.Conv2d(2*32,32, 3, 1, 1))
        self.BN3 = nn.BatchNorm2d(32)

        self.deconv4 = spectral_norm(nn.Conv2d(32, 1, 2, stride=2))


    def forward(self, input):
        _ = self.fc1(input)
        _ = _.view((-1,1,56,56))
        _ = self.relu(self.BN1(_))
        _ = self.relu(self.BN2(self.deconv2(_)))
        _ = self.relu(self.BN3(self.deconv3(_)))
        _ = self.tanh(self.deconv4(_))

        return _

class Discriminator_MNIST(Module):
    def __init__(self):
        super(Discriminator_MNIST, self).__init__()

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 32, 5, padding=2)),  # batch, 32, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)
            # nn.Conv2d(32, 32, 3, stride=2,padding=1),  # batch, 32, 14, 14
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(32, 64, 5, padding=2)),  # batch, 64, 14, 14
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)
            # nn.Conv2d(64, 64, 3, stride=2,padding=1)  # batch, 64, 7, 7
        )
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(64 * 7 * 7, 1024)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Linear(1024, 1)),
            # nn.Sigmoid()
        )

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        # reshape
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class LeNet(Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        out = self.fc3(y)
        return out, y


opt_cuda = True
FloatTensor = torch.cuda.FloatTensor if opt_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if opt_cuda else torch.LongTensor

def trainGAN(args, netG, netD, dataloader, optimizerG, optimizerD, totaliter, verbose, device):
    if verbose:
        if os.path.exists(os.path.join(args.path,'sample')) is False:
            os.makedirs(os.path.join(args.path,'sample'))

    if not verbose:
        tqdm = lambda x: x

    crit = nn.BCEWithLogitsLoss()
    xxz = Variable(torch.randn(80, 128, device=device))
    t_iter = 0
    for epoch in range(9999999):
        disc_loss_set = []
        gen_loss_set = []
        prog_bar = tqdm(dataloader)
        for _, (data, _) in enumerate(prog_bar):
            batch_size = data.size(0)
            if batch_size!=0:

                ones = Variable(torch.ones(batch_size, 1, device=device))
                zeros = Variable(torch.zeros(batch_size, 1, device=device))
            
                data = Variable(data.cuda())

                # update discriminator
                z = Variable(torch.randn(batch_size, 128, device=device))
                optimizerD.zero_grad()
                optimizerG.zero_grad()

                fake = netG(z)
                fake = DiffAugment(fake, policy=policy)
                data = DiffAugment(data, policy=policy)

                disc_loss = crit(netD(data), ones) + crit(netD(fake), zeros)
                disc_loss.backward()
                optimizerD.step()

                z = Variable(torch.randn(batch_size, 128, device=device))

                # update generator
                optimizerD.zero_grad()
                optimizerG.zero_grad()
                # FS_netD.eval()

                fake = netG(z)
                fake = DiffAugment(fake, policy=policy)

                gen_loss = crit(netD(fake), ones)
                
                gen_loss.backward()
                optimizerG.step()
                disc_loss_set.append(disc_loss.item())
                gen_loss_set.append(gen_loss.item())

                prog_bar.set_description(f'iter {t_iter} disc loss {np.array(disc_loss_set).mean():.5f} gen loss {np.array(gen_loss_set).mean():.5f}')

                if t_iter % 1000 == 0 and verbose:
                    print(epoch, ' disc loss', np.array(disc_loss_set).mean(), 'gen loss', np.array(gen_loss_set).mean())
                    fake = netG(xxz)
                    vutils.save_image(
                        (fake.data+1)/2,
                        f'{args.path}/sample/sample_{t_iter:03d}.png',
                        nrow=10
                    )

                t_iter += 1
                if t_iter >= totaliter:
                    break
        if t_iter >= totaliter:
            break
        

    return netG, netD

def trainSubGAN(args, netG, netSubD, netM, dataloader, optimizerSubD, optimizerM, totaliter, verbose, device):
    if verbose:
        if os.path.exists(os.path.join(args.path,'sample')) is False:
            os.makedirs(os.path.join(args.path,'sample'))

    if not verbose:
        tqdm = lambda x: x

    crit = nn.BCEWithLogitsLoss()
    crit.cuda()
    xxz = Variable(torch.randn(80, 128, device=device))
    t_iter = 0
    for epoch in range(999999):
        disc_loss_set = []
        gen_loss_set = []
        prog_bar = tqdm(dataloader)
        for i, data in enumerate(prog_bar):
            real_image, _ = data
            batch_size = real_image.size(0)

            ones = Variable(torch.ones(batch_size, 1).type(FloatTensor))
            zeros = Variable(torch.zeros(batch_size, 1).type(FloatTensor))

            # Configure input
            real_image = Variable(real_image.type(FloatTensor).type(FloatTensor))

            # update discriminator
            z = Variable(torch.randn(batch_size, args.nz).type(FloatTensor))
            optimizerSubD.zero_grad()
            optimizerM.zero_grad()

            sub_z = netM(z)
            fake = netG(sub_z)
            fake = DiffAugment(fake, policy=policy)
            real_image = DiffAugment(real_image, policy=policy)
            
            disc_loss = crit(netSubD(real_image), ones) + crit(netSubD(fake), zeros)
            disc_loss.backward()
            optimizerSubD.step()

            z = Variable(torch.randn(batch_size, args.nz).type(FloatTensor))


            # update generator
            optimizerSubD.zero_grad()
            optimizerM.zero_grad()

            sub_z = netM(z)
            fake = netG(sub_z)
            fake = DiffAugment(fake, policy=policy)


            gen_loss = crit(netSubD(fake), ones)
            gen_loss.backward()
            optimizerM.step()
            disc_loss_set.append(disc_loss.item())
            gen_loss_set.append(gen_loss.item())

            prog_bar.set_description(f'iter {t_iter} disc loss {np.array(disc_loss_set).mean():.5f} gen loss {np.array(gen_loss_set).mean():.5f}')
            
            if t_iter % 1000 == 0 and verbose:
                print(epoch, ' disc loss', np.array(disc_loss_set).mean(), 'gen loss', np.array(gen_loss_set).mean())
                fake = netG(xxz)
                vutils.save_image(
                    (fake.data+1)/2,
                    f'{args.path}/sample/sample_{t_iter:03d}.png',
                    nrow=10
                )
                
            t_iter += 1
            if t_iter >= totaliter:
                break
        if t_iter >= totaliter:
            break


    return netG, netSubD