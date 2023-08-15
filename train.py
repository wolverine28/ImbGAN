from __future__ import print_function

import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
from torch.autograd import Variable

from networks.MNIST_networks import (Discriminator_MNIST, Generator_MNIST,
                                     LeNet, trainGAN)
from utils import MyDataset, compute_BA, make_Imb, weights_init, StratifiedBatchSampler

# parsers
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='MNIST')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--path', required=True, help='path to save pre-trained GAN model')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--nepoch', type=int, default=60, help='number of epoch to train for')
parser.add_argument('--niter', type=int, default=1000, help='number of epoch to train for GAN for each epoch of ImbGAN')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--IR', type=float, default=0.01, help='imbalance ratio')
parser.add_argument('--OR', type=str, default='0,1', help='Overlapped classes')
parser.add_argument('--beta', type=float, default=0.5, help='Overlapped classes')
parser.add_argument('--pre_train_path', type=str, default=True, help='path of pre-trained GAN')
parser.add_argument('--manualSeed', type=int, default=8888, help='seed')
args = parser.parse_args()

FloatTensor = torch.cuda.FloatTensor
def eval(netC, dataloader_test, epoch):
    netC.eval()
    test_err = []
    outputs = []
    targets = []
    with torch.no_grad():
        for data,target in dataloader_test:
            # Configure input
            data = Variable(data.type(FloatTensor))
            target = Variable(target.type(FloatTensor))
            output,_ = netC(data)

            predicted = torch.sigmoid(output.data)>0.5

            outputs.append(predicted.cpu().numpy())
            targets.append(target.cpu().numpy())

    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)

    print('             epoch:{:04d}  test_BA:{:.3f}'.format(epoch, compute_BA(torch.tensor(outputs)[:,0],torch.tensor(targets))))

def train_ImbGAN(args,imb_dataset,dataloader, dataloader_test):
    try:
        os.makedirs(args.path)
    except OSError:
        pass

    name = f'{args.dataset}_{args.IR}_{args.OR}_{args.manualSeed}'

    # Define the generator and initialize the weights
    netG = Generator_MNIST(args.nz)
    netG.load_state_dict(torch.load(os.path.join(args.path,f'{name}_G.ckpt')))
    netG.cuda()

    # Define the discriminator and initialize the weights
    netD = Discriminator_MNIST()
    netD.load_state_dict(torch.load(os.path.join(args.path,f'{name}_D.ckpt')))
    netD.cuda()

    netC = LeNet()
    criterion  = nn.BCEWithLogitsLoss()

    netC.cuda()
    criterion.cuda()
    netC.apply(weights_init)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0, 0.9))
    optimizer = optim.Adam(netC.parameters(),lr=args.lr)
    criterion = nn.BCEWithLogitsLoss().cuda()


    z = Variable(FloatTensor(np.random.normal(0, 1, (10000, 128))))
    m_prime = Variable(netG(z).data)

    z = Variable(FloatTensor(np.random.normal(0, 1, (10000, 128))))
    m_prime_miss = Variable(netG(z).data)

    m_miss = imb_dataset[imb_dataset.target==1][0].cuda()

    for e in range(args.nepoch):
        running_loss = []
        running_acc = []

        all_miss = []
        netC.train()
        for data, target in dataloader:
            data = Variable(data.type(FloatTensor))
            target = Variable(target.type(FloatTensor))

            ## Prepare training set
            ID = int(((target==0).sum()-(target==1).sum()).item()*1)
            if ID<=0:
                ID=0
            ID_origin = int(ID*(1-args.beta))
            ID_overlap = ID-ID_origin
            t_m_prime = m_prime[np.random.choice(range(m_prime.shape[0]),ID_origin)]
            t_m_prime_miss = m_prime_miss[np.random.choice(range(m_prime_miss.shape[0]),ID_overlap)]
            syn_data = torch.cat((t_m_prime,t_m_prime_miss))

            all_data = torch.cat((data,syn_data))
            all_target = torch.cat((target,torch.ones(syn_data.shape[0]).cuda()))

            ## Update Classifier
            optimizer.zero_grad()   # zero the gradient buffers
            output, _ = netC(all_data)
            loss = criterion(output.view(-1), all_target)
            loss.backward()
            optimizer.step()    # Does the update

            ## Evaluation
            output, _ = netC(data)
            predicted = (torch.sigmoid(output.data)>0.5).view(-1)
            running_loss.append(loss.item())
            BA = compute_BA(predicted,target)
            running_acc.append(BA.item()) 

            ## Update miss set
            all_miss.append(data[torch.logical_and(target==1,predicted==False)])

        batch_miss = torch.cat(all_miss)
        alpha = (1+np.cos((e)/args.nepoch*np.pi))*0.5
        l = np.random.choice(range(batch_miss.shape[0]),int(np.ceil(batch_miss.shape[0]*alpha)), replace=False)
        idx = np.random.choice(range(m_miss.shape[0]),len(l), replace=False)
        m_miss[idx] = batch_miss[l]

        batch_loss = np.mean(running_loss)
        batch_acc = np.mean(running_acc)
        print('epoch:{:04d}  loss:{:.3f}  BA:{:.3f}'.format(e,batch_loss,batch_acc))


        ## Update Generator
        m_miss_dataset = MyDataset(m_miss,torch.ones(m_miss.shape[0]))
        m_miss_dataloader = torch.utils.data.DataLoader(m_miss_dataset, batch_size=128,
                                            shuffle=True, num_workers=int(0))
        print('Training GAN model')             

        netG, netD = trainGAN(args, netG, netD, m_miss_dataloader, optimizerG, optimizerD, args.niter, False, 'cuda:0')

        z = Variable(FloatTensor(np.random.normal(0, 1, (10000, 128))))
        m_prime_miss = Variable(netG(z).data)
        netG.load_state_dict(torch.load(os.path.join(args.path,f'{name}_G.ckpt')))
        netD.load_state_dict(torch.load(os.path.join(args.path,f'{name}_D.ckpt')))

        eval(netC, dataloader_test, args.nepoch)

    eval(netC, dataloader_test, args.nepoch)

if __name__=='__main__':
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)
    np.random.seed(args.manualSeed)
    name = f'{args.dataset}_{args.IR}_{args.OR}_{args.manualSeed}'

    with open(os.path.join(args.path,f'{name}_imb_dataset.pkl'),'rb') as f:
        imb_dataset = pickle.load(f)

    if args.dataset == 'MNIST':
        dataset = dset.MNIST
    else:
        raise NotImplementedError
    
    # load data
    sampler = StratifiedBatchSampler(imb_dataset.target, batch_size=args.batchSize)
    dataloader = torch.utils.data.DataLoader(imb_dataset, batch_sampler=sampler,
                                            shuffle=False, num_workers=int(args.workers))

    imb_dataset_test = make_Imb(dataset,[0,1,2,3,4],args.IR,train=False,normalize=True)
    dataloader_test = torch.utils.data.DataLoader(imb_dataset_test, batch_size=args.batchSize,
                                            shuffle=True, num_workers=int(args.workers))

    # train ImbGAN
    train_ImbGAN(args,imb_dataset,dataloader, dataloader_test)