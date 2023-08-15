import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Sampler, Dataset

opt_dataroot = './dataset'



# custom weights initialization called on netG and netD
def weights_init(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)

# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc


def compute_BA(preds, labels):

    TP = torch.logical_and(preds==1,labels==1).sum()
    FP = torch.logical_and(preds==1,labels==0).sum()
    TN = torch.logical_and(preds==0,labels==0).sum()
    FN = torch.logical_and(preds==0,labels==1).sum()

    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)

    # if target.sum()!=0:

    BA = (TPR+TNR)/2
    return BA


class MyDataset_origin(Dataset):

    def __init__(self, data, target, original_target):
        self.data = data
        self.target = target
        self.original_target = original_target

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # return {"input":self.data[idx,:,:], 
        #         "label": self.target[idx]}

        return (self.data[idx,:,:,:],self.target[idx],self.original_target[idx])

class MyDataset(Dataset):

    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # return {"input":self.data[idx,:,:], 
        #         "label": self.target[idx]}

        return (self.data[idx,:,:,:],self.target[idx])

def make_Imb(dset_name,minor_cls_index,IR,train,normalize,need_origin=False):
    t_normalize=transforms.Normalize((0.0), (1.0,))
    if normalize:
        t_normalize=transforms.Normalize((0.5), (0.5,))
    
    # t_normalize=transforms.Normalize((0.0,0.0,0.0,), (1.0,1.0,1.0,))
    # if normalize:
    #     t_normalize=transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))

    ds = dset_name(
    root=opt_dataroot, download=True,train=train,
    transform=transforms.Compose([
        transforms.ToTensor(),
        # Per_pixcel_subtract(train_mean),
        t_normalize,

    ]))
    original_targets = ds.targets

    minor_bool = np.in1d(original_targets,minor_cls_index)
    modified_targets = list(minor_bool*1)

    minor_idx = np.where(np.array(modified_targets)==1)[0]
    major_idx = np.where(np.array(modified_targets)==0)[0]
    selected_minor_idx = np.random.choice(minor_idx,int(len(major_idx)*IR))
    selected_idx = np.concatenate((selected_minor_idx,major_idx))


    data = torch.cat([ds[i][0] for i in selected_idx])
    data = data[:,None,:,:]
    modified_targets = np.array(modified_targets)[selected_idx]
    original_targets = original_targets[selected_idx]

    if need_origin:
        imb_data = MyDataset_origin(data,modified_targets,original_targets)
    else:
        imb_data = MyDataset(data,modified_targets)

    return imb_data 

class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)



# def overlapped_imb_data(netEnc,netDec,dset_name,OR,IR,opt_batchSize,opt_workers=0,opt_cuda=True,normalize=False):
#     FloatTensor = torch.cuda.FloatTensor if opt_cuda else torch.FloatTensor  

#     imb_dataset = make_Imb(dset_name,[0,1,2,3,4],IR,train=True,normalize=normalize)

#     y = torch.from_numpy(np.array(imb_dataset.dataset.targets)[imb_dataset.indices])
#     sampler = StratifiedBatchSampler(y, batch_size=opt_batchSize)                        
#     dataloader = torch.utils.data.DataLoader(imb_dataset, batch_size=5120,
#                                     shuffle=True, num_workers=int(opt_workers))

#     ##############################################################################
#     additive_data = []
#     for data,target in dataloader:
#         data = data.cuda()
#         target = target.cuda()

#         latent = netEnc(data)

#         ov_data,ov_target = do_overlap(latent.cpu().data.numpy(),target.cpu().data.numpy(),target_label=1,k1=3,k2=7,overlap_ratio=OR)
#         # latent = latent+0.02
#         if ov_data is not None:
#             gen_image = netDec(FloatTensor(ov_data))

#             additive_data.append(gen_image)

#     additive_data = torch.cat(additive_data,dim=0)
#     additive_data = additive_data[:,0,:,:]
#     additive_target = torch.zeros(additive_data.shape[0])

    
#     y = torch.from_numpy(np.array(imb_dataset.dataset.targets)[imb_dataset.indices])
#     del_idx = np.random.choice(torch.where(y==0)[0],additive_data.shape[0])
#     selected_idx = np.setdiff1d(np.arange(len(imb_dataset.indices)),del_idx)

#     origin_data = torch.cat([imb_dataset[i][0] for i in selected_idx],dim=0)
#     origin_target = [imb_dataset[i][1] for i in selected_idx]
    
#     data = torch.cat((additive_data.cpu(),origin_data))
#     data = data[:,None,:,:]
#     label = np.concatenate((additive_target,origin_target))

#     ##############################################################################
#     mydataset = MyDataset(data,label)
#     sampler = StratifiedBatchSampler(mydataset.target, batch_size=opt_batchSize)

#     dataloader = torch.utils.data.DataLoader(mydataset, batch_sampler=sampler,
#                                             shuffle=False, num_workers=int(opt_workers))
    
#     return mydataset,dataloader




def new_overlapped_imb_data(dset_name,OR,IR,opt_batchSize,normalize=False):
    opt_workers = 0

    imb_dataset = make_Imb(dset_name,[0,1,2,3,4],IR,train=True,normalize=normalize,need_origin=True)
    
    additive_data = []
    for o in (OR.replace(',','')):
        o = int(o)
        idx = np.where(imb_dataset.original_target==o)[0]
        iidx = np.random.choice(idx,int(len(idx)*0.5),replace=True)
        additive_data.append(imb_dataset.data[iidx])

    additive_data = torch.cat(additive_data,dim=0)
    additive_target = torch.zeros(additive_data.shape[0])

    ##############################################################################

    data = torch.cat((additive_data.cpu(),imb_dataset.data))
    label = np.concatenate((additive_target,imb_dataset.target))

    ##############################################################################
    mydataset = MyDataset(data,label)
    sampler = StratifiedBatchSampler(mydataset.target, batch_size=opt_batchSize)

    dataloader = torch.utils.data.DataLoader(mydataset, batch_sampler=sampler,
                                            shuffle=False, num_workers=int(opt_workers))
    
    return mydataset,dataloader