import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

import models
from data.datasets import GetVMTdata, GetCINEdata
from models.UnetGenerator import OnlyUp

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-training_model', '--training_model', default="VMT", type=str,
                    help='Train VMT or CINE')
parser.add_argument('-batch_size', '--batch_size', default=1, type=int,
                    help='Batch size')
parser.add_argument('-epoch', '--epoch', default=100, type=int,
                    help='Epoch')
parser.add_argument('-save_frq', '--save_frq', default=10, type=int,
                    help='Save frequency')
parser.add_argument('-workers', '--workers', default=0, type=int,
                    help='Workers')
parser.add_argument('-pretrain', '--pretrain', default="", type=str,
                    help='Pretrain weights path')
parser.add_argument('-data_root', '--data_root', required=True, type=str,
                    help='Data root')
parser.add_argument('-train_txt', '--train_txt', required=True, type=str,
                    help='Training list')
parser.add_argument('-valid_txt', '--valid_txt', required=True, type=str,
                    help='Validation list')

# Parse arguments
args = parser.parse_args()

training_model = args.training_model
device = (f'cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = args.batch_size
eps = 1e-5
seed = 1024
num_trainG = 1
num_epochs = args.epoch
save_frq = args.save_frq
criterionGAN = nn.MSELoss()
criterionL1 = nn.L1Loss()
real_label=torch.tensor(1)
fake_label=torch.tensor(0)
lambda_L1 = 100.0
Network_G = getattr(models, 'UnetGenerator')
Network_D = getattr(models, 'NLayerDiscriminator')
lr = 1e-4
weight_decay = 1e-5
amsgrad = True
workers = args.workers
pretrain_path = args.pretrain
data_root = args.data_root
train_txt = args.train_txt
valid_txt = args.valid_txt

print("training_model: ", training_model)
print("device: ", device)
print("batch_size: ", batch_size)
print("eps: ", eps)
print("num_trainG: ", num_trainG)
print("num_epochs: ", num_epochs)
print("save_frq: ", save_frq)
print("lr: ", lr)
print("weight_decay: ", weight_decay)
print("workers: ", workers)
print("pretrain_path: ", pretrain_path)
print("data_root: ", data_root)
print("train_txt: ", train_txt)
print("valid_txt: ", valid_txt)

def main():
    if device != 'cpu': os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if not os.path.exists('./ckpts'):
        os.makedirs('./ckpts')
    if not os.path.exists('./training_process'):
        os.makedirs('./training_process')

    # Build model
    if training_model == 'VMT':
        if pretrain_path:
            model_G = Network_G(in_channels = 1, out_channels = 2).to(device)
            model_G.load_state_dict(torch.load(pretrain_path))
            model_G.up63 = OnlyUp(24, 7).to(device)
        else:
            model_G = Network_G(in_channels = 1, out_channels = 7).to(device)
        model_D = Network_D(input_nc = 8).to(device)
        
        # Load VMT data
        train_set = GetVMTdata(train_txt, data_root)
        valid_set = GetVMTdata(valid_txt, data_root)
        
    else:
        model_G = Network_G(in_channels = 1, out_channels = 2).to(device)
        model_D = Network_D(input_nc = 3).to(device)
        
        # Load CINE data
        train_set = GetCINEdata(train_txt, data_root)
        valid_set = GetCINEdata(valid_txt, data_root)
        
    model_VGG = torchvision.models.vgg19(pretrained=True).to(device)
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr = lr, weight_decay= weight_decay, amsgrad = amsgrad)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr = lr, weight_decay= weight_decay, amsgrad = amsgrad)

    train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle = True,
            num_workers=workers)

    valid_loader = DataLoader(
            dataset=valid_set,
            batch_size=1,
            shuffle = False,
            num_workers=workers)

    tGloss_plt = []
    tDloss_plt = []
    tGlosses = AverageMeter()
    tDlosses = AverageMeter()
    
    vGloss_plt = []
    vDloss_plt = []
    vGlosses = AverageMeter()
    vDlosses = AverageMeter()

    # Start training
    for j in range(num_epochs):
        print(f"Epoch:{j+1}/{num_epochs}")
        
        # Training
        print('Train')
        model_G.train()
        model_D.train()
        for data in tqdm(train_loader):
            adjust_learning_rate(optimizer_G, j+1, num_epochs, lr)
            adjust_learning_rate(optimizer_D, j+1, num_epochs, lr)

            real_A = data[0].to(device)
            real_B = data[1].to(device)
            
            for t in range(num_trainG):
                fake_B = model_G(real_A)
                
                optimizer_G.zero_grad()
                # First, G(A) should fake the discriminator
                fake_AB = torch.cat((real_A, fake_B), 1)
                pred_fake = model_D(fake_AB)
                target_tensor = real_label
                target_tensor = target_tensor.expand_as(pred_fake).to(device)
                loss_G_GAN = criterionGAN(pred_fake.to(torch.float32), target_tensor.to(torch.float32))
                # Second, G(A) = B
                loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1 
                # Third, Feature(G(A)) = Feature(B)
                loss_G_feature = 0
                for i in range(7 if training_model == 'VMT' else 2):
                    loss_G_feature += criterionGAN(model_VGG(torch.cat([fake_B[:,i:i+1,...], fake_B[:,i:i+1,...], fake_B[:,i:i+1,...]], dim=1)), model_VGG(torch.cat([real_B[:,i:i+1,...], real_B[:,i:i+1,...], real_B[:,i:i+1,...]], dim=1)))
                
                # combine loss and calculate gradients
                loss_G = loss_G_GAN + loss_G_L1 + loss_G_feature
                loss_G.backward()
                optimizer_G.step()
                
            
            optimizer_D.zero_grad()
            # Fake; stop backprop to the generator by detaching fake_B
            fake_AB = torch.cat((real_A, fake_B), 1) 
            pred_fake = model_D(fake_AB.detach())
            target_tensor = fake_label
            target_tensor = target_tensor.expand_as(pred_fake).to(device)
            loss_D_fake = criterionGAN(pred_fake.to(torch.float32), target_tensor.to(torch.float32))
            # Real
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = model_D(real_AB)
            target_tensor = real_label
            target_tensor = target_tensor.expand_as(pred_real).to(device)
            loss_D_real = criterionGAN(pred_real.to(torch.float32), target_tensor.to(torch.float32))

            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            tGlosses.update(loss_G.item(), batch_size)
            tDlosses.update(loss_D.item(), batch_size)

        print()
        print('-----------------------Train-----------------------')
        print('G Loss {:.7f}'.format(tGlosses.avg))
        print('D Loss {:.7f}'.format(tDlosses.avg))
        print('---------------------------------------------------')

        tGloss_plt.append(tGlosses.avg)
        tDloss_plt.append(tDlosses.avg)
        tGlosses.reset()
        tDlosses.reset()

        # Validation
        print('Vaild')
        model_G.eval()
        model_D.eval()
        for i, data in enumerate(tqdm(valid_loader)):
            real_A = data[0].to(device)
            real_B = data[1].to(device)
            
            fake_B = model_G(real_A)
                
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = model_D(fake_AB)
            target_tensor = real_label
            target_tensor = target_tensor.expand_as(pred_fake).to(device)
            loss_G_GAN = criterionGAN(pred_fake.to(torch.float32), target_tensor.to(torch.float32))
            # Second, G(A) = B
            loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1
            # Third, Feature(G(A)) = Feature(B)
            loss_G_feature = 0
            for k in range(7 if training_model == 'VMT' else 2):
                loss_G_feature += criterionGAN(model_VGG(torch.cat([fake_B[:,k:k+1,...], fake_B[:,k:k+1,...], fake_B[:,k:k+1,...]], dim=1)), model_VGG(torch.cat([real_B[:,k:k+1,...], real_B[:,k:k+1,...], real_B[:,k:k+1,...]], dim=1)))
            
            # combine loss and calculate gradients
            loss_G = loss_G_GAN + loss_G_L1 + loss_G_feature
            
            # Fake; stop backprop to the generator by detaching fake_B
            fake_AB = torch.cat((real_A, fake_B), 1) 
            pred_fake = model_D(fake_AB.detach())
            target_tensor = fake_label
            target_tensor = target_tensor.expand_as(pred_fake).to(device)
            loss_D_fake = criterionGAN(pred_fake.to(torch.float32), target_tensor.to(torch.float32))
            # Real
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = model_D(real_AB)
            target_tensor = real_label
            target_tensor = target_tensor.expand_as(pred_real).to(device)
            loss_D_real = criterionGAN(pred_real.to(torch.float32), target_tensor.to(torch.float32))

            loss_D = (loss_D_fake + loss_D_real) * 0.5

            if (j+1) % (save_frq/2) == 0:
                if not os.path.exists(f'./valid_{j+1}'):
                    os.makedirs(f'./valid_{j+1}')
                    
                if training_model == 'VMT':
                    name = valid_set.path[i].split("/")[-1].split(".npz")[0]
                    for k in range(7):
                        if not os.path.exists(f'./valid_{j+1}/{k}'):
                            os.makedirs(f'./valid_{j+1}/{k}')
                        plt.figure()
                        plt.axis('off')
                        plt.imshow(np.squeeze(fake_B.to('cpu').detach().numpy())[k, ...], cmap='gray', vmin = 0, vmax = 1)
                        plt.savefig(f"./valid_{j+1}/{k}/{name}_G.jpg")

                        plt.figure()
                        plt.axis('off')
                        plt.imshow(np.squeeze(real_B.to('cpu').detach().numpy())[k, ...], cmap='gray', vmin = 0, vmax = 1)
                        plt.savefig(f"./valid_{j+1}/{k}/{name}_real.jpg")

                        plt.close('all')

                else:
                    name = valid_set.cine_path[i].split("/")[-1].split(".mat")[0]
                    plt.figure()
                    plt.imshow(np.squeeze(real_A[:,0:1,...].to('cpu').detach().numpy()), cmap="gray", vmax=1, vmin=0)
                    plt.axis("off")
                    plt.savefig(f"./valid_{j+1}/{name}_input.jpg")
                    
                    plt.figure()
                    plt.imshow(np.squeeze(fake_B[:,0:1,...].to('cpu').detach().numpy()), cmap="gray", vmax=1, vmin=0)
                    plt.axis("off")
                    plt.savefig(f"./valid_{j+1}/{name}_imG.jpg")
                    
                    plt.figure()
                    plt.imshow(np.squeeze(real_B[:,0:1,...].to('cpu').detach().numpy()), cmap="gray", vmax=1, vmin=0)
                    plt.axis("off")
                    plt.savefig(f"./valid_{j+1}/{name}_imreal.jpg")
                    
                    plt.figure()
                    plt.imshow(np.squeeze(fake_B[:,1:2,...].to('cpu').detach().numpy()), cmap="gray", vmax=1, vmin=0)
                    plt.axis("off")
                    plt.savefig(f"./valid_{j+1}/{name}_maskG.jpg")
                    
                    plt.figure()
                    plt.imshow(np.squeeze(real_B[:,1:2,...].to('cpu').detach().numpy()), cmap="gray", vmax=1, vmin=0)
                    plt.axis("off")
                    plt.savefig(f"./valid_{j+1}/{name}_maskreal.jpg")
                    
                    plt.close('all')
                
            vGlosses.update(loss_G.item(), batch_size)
            vDlosses.update(loss_D.item(), batch_size)

        print()
        print('-----------------------Valid-----------------------')
        print('G Loss {:.7f}'.format(vGlosses.avg))
        print('D Loss {:.7f}'.format(vDlosses.avg))
        print('---------------------------------------------------')

        vGloss_plt.append(vGlosses.avg)
        vDloss_plt.append(vDlosses.avg)
        vGlosses.reset()
        vDlosses.reset()

        print()

        if ((j+1) % save_frq == 0 ) & (j != 1):
            plot((j+1), tGloss_plt, tDloss_plt, vGloss_plt, vDloss_plt)
            file_name = os.path.join(f'./ckpts/modelG_epoch_{j+1}.pt')
            torch.save(model_G.state_dict(),file_name)
            file_name = os.path.join(f'./ckpts/modelD_epoch_{j+1}.pt')
            torch.save(model_D.state_dict(),file_name)
            
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val_all = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val_all.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.std = np.std(np.array(self.val_all))

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def plot(epoch, tGloss_plt, tDloss_plt, vGloss_plt, vDloss_plt):
    plt.figure()
    plt.plot(tGloss_plt,'-', label='Train')
    plt.plot(vGloss_plt,'-', label='Valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('G Train vs Valid Loss')
 
    plt.savefig(f'./training_process/epoch{epoch} G Train vs Valid Loss.png')  

    plt.figure()
    plt.plot(tDloss_plt,'-', label='Train')
    plt.plot(vDloss_plt,'-', label='Valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('D Train vs Valid Loss')
 
    plt.savefig(f'./training_process/epoch{epoch} D Train vs Valid Loss.png')  

    plt.close('all')

if __name__ == '__main__':
    main()





