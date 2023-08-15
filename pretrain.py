import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset

from networks.MNIST_networks import (Discriminator_MNIST, Generator_MNIST,
                                     trainGAN)
from utils import make_Imb, new_overlapped_imb_data

# parsers
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='MNIST')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--path', required=True, help='path to save pre-trained GAN model')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=6000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--IR', type=float, default=0.01, help='imbalance ratio')
parser.add_argument('--OR', type=str, default='0,1', help='Overlapped classes')
parser.add_argument('--prepretrain', type=bool, default=True, help='If IR is too small,\
                     train GAN with majority classes and fine-tune to the minority classes')
parser.add_argument('--pre_pre_train_path', type=str, default="./ckpts", help='path to save pre-pre-trained GAN')
parser.add_argument('--manualSeed', type=int, default=8888, help='seed')
args = parser.parse_args()


#########################################################################################

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
device = 'cuda:0'

if args.path is not None:
    if os.path.exists(args.path) is False:
        os.makedirs(args.path)

if __name__ == '__main__':
    name = f'{args.dataset}_{args.IR}_{args.OR}_{args.manualSeed}'

    # Define the generator and discriminator
    netG = Generator_MNIST(args.nz).to(device)
    optimizerG = optim.Adam(netG.parameters(),lr=args.lr)

    netD = Discriminator_MNIST().to(device)
    optimizerD = optim.Adam(netD.parameters(),lr=args.lr)

    if args.dataset == 'MNIST':
        dataset = dset.MNIST
    else:
        raise NotImplementedError
    

    imb_dataset, _ = new_overlapped_imb_data(dset.MNIST,args.OR,args.IR,args.batchSize,normalize=True)
    with open(os.path.join(args.path,f'{name}_imb_dataset.pkl'),'wb') as f:
        pickle.dump(imb_dataset,f)

    if args.pre_pre_train_path is not None:
        if os.path.exists(args.pre_pre_train_path) is False:
            os.makedirs(args.pre_pre_train_path)

    if args.prepretrain and args.pre_pre_train_path is not None:
        dataloader = torch.utils.data.DataLoader(imb_dataset, batch_size=args.batchSize, 
                                                shuffle=True, num_workers=int(args.workers))

        netG, netD = trainGAN(args, netG, netD, dataloader, optimizerG, optimizerD, args.niter//2+1, True, device)
        torch.save(netG.state_dict(),os.path.join(args.pre_pre_train_path,f'{name}_prepreG.ckpt'))
        torch.save(netD.state_dict(),os.path.join(args.pre_pre_train_path,f'{name}_prepreD.ckpt'))


    if args.prepretrain and args.pre_pre_train_path is not None:
        netG.load_state_dict(torch.load(os.path.join(args.pre_pre_train_path,f'{name}_prepreG.ckpt')))
        netD.load_state_dict(torch.load(os.path.join(args.pre_pre_train_path,f'{name}_prepreD.ckpt')))

        for ct, child in enumerate(netD.children()):
            print(ct,child)
            if ct<2:
                for param in child.parameters():
                    param.requires_grad = False


    min_data = torch.utils.data.Subset(imb_dataset,np.where(imb_dataset.target==1)[0])
    dataloader = torch.utils.data.DataLoader(min_data, batch_size=args.batchSize, 
                                            shuffle=True, num_workers=int(args.workers))
    netG, netD = trainGAN(args, netG, netD, dataloader, optimizerG, optimizerD, args.niter, True, device)
    torch.save(netG.state_dict(),os.path.join(args.path,f'{name}_G.ckpt'))
    torch.save(netD.state_dict(),os.path.join(args.path,f'{name}_D.ckpt'))

####

