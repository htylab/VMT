import os
import numpy as np
from tqdm import tqdm

import torch
import torch.optim
from torch.utils.data import DataLoader

import models
from data.datasets import GetVMTdata

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-workers', '--workers', default=0, type=int,
                    help='Workers')
parser.add_argument('-weights_path', '--weights_path', required=True, type=str,
                    help='Weights path')
parser.add_argument('-data_root', '--data_root', required=True, type=str,
                    help='Data root')
parser.add_argument('-valid_txt', '--valid_txt', required=True, type=str,
                    help='Validation list')
parser.add_argument('-save_path', '--save_path', required=True, type=str,
                    help='Save result path')

# Parse arguments
args = parser.parse_args()

device = (f'cuda:0' if torch.cuda.is_available() else 'cpu')
Network_G = getattr(models, 'UnetGenerator')
workers = args.workers
weights_path = args.weights_path
data_root = args.data_root
valid_txt = args.valid_txt
save_path = args.save_path

print("device: ", device)
print("workers: ", workers)
print("weights_path: ", weights_path)
print("data_root: ", data_root)
print("valid_txt: ", valid_txt)
print("save_path: ", save_path)

def main():
    if device != 'cpu': os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Build model
    model_G = Network_G(in_channels = 1, out_channels = 7).to(device)
    model_G.load_state_dict(torch.load(args.weights_path))

    # Load data
    valid_set = GetVMTdata(valid_txt, data_root)

    valid_loader = DataLoader(
            dataset=valid_set,
            batch_size=1,
            shuffle = False,
            num_workers=workers)
    
    model_G.eval()
    
    #Produce VMT
    for i, data in enumerate(tqdm(valid_loader)):
        real_A = data[0].to(device)
        
        fake_B = np.squeeze(model_G(real_A).to('cpu').detach().numpy())
        fake_im = np.squeeze(fake_B)
        
        file = np.load(valid_set.path[i])
        im = file['im']
        
        name = valid_set.path[i].split("/")[-1].split(".npz")[0]
        
        # Restore magnitude
        for j in range(im.shape[0]-1):
            fake_im[j, ...] = fake_im[j, ...] * im[j, ...].max()
        
        np.savez(f'{save_path}/{name}.npz', im=im, target_im=fake_im, invtime=file['invtime'])
        
if __name__ == '__main__':
    main()