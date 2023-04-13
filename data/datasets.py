import numpy as np
import os
import torch
import cv2

class GetVMTdata():
    def __init__(self, list_file, root = ''):
        self.path = []
        with open(list_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                self.path.append(os.path.join(root, line))
        
    def __len__(self):
        return len(self.path)    
    
    def __getitem__(self, idx):
        file = np.load(self.path[idx])
        
        im = file['im']
        
        for i in range(im.shape[0]):
            im[i, ...] = (im[i, ...]/im[i, ...].max())
        
        input = torch.from_numpy(im[0:1, ...].astype(np.float32))
        output = torch.from_numpy(im[0:-1, ...].astype(np.float32))
    
        return input, output

class GetCINEdata():
    def __init__(self, list_cinefile, root = ''):
        self.cine_path = []
        with open(list_cinefile, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                self.cine_path.append(os.path.join(root, line))
        
    def __len__(self):
        return len(self.cine_path)    
    
    def __getitem__(self, idx):
        cine = np.load(self.cine_path[idx])
        
        input = cine['image']
        im = input.copy()
        mask = cine['mask']
        
        input = input/input.max()
        
        im_eq = (im/im.max()*255).astype(np.uint8)
        im_eq = cv2.equalizeHist(im_eq)
        im_eq = im_eq/im_eq.max()
        
        mask = mask/mask.max()
        
        input = input.astype(np.float32)
        im_eq = im_eq.astype(np.float32)
        mask = mask.astype(np.float32)
        
        input = np.stack([input])
        im_eq = np.stack([im_eq])
        mask = np.stack([mask])
        
        input = torch.from_numpy(input)
        im_eq = torch.from_numpy(im_eq)
        mask = torch.from_numpy(mask)
        
        output = torch.cat([im_eq, mask], dim=0)

        return input, output