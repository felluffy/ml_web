import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import matplotlib.animation as manim
import os
import rawpy
import skimage.measure as sK_measure
from torch.utils.data import DataLoader
from torchvision import transforms
import time

cuda = torch.device('cuda')
batch_size = 8
ds_folder = './Sony/'
try: 
    os.makedirs(ds_folder)
except Exception as e:
    pass
ds_exists = False
ds_train_sub = 'sony'
# for folder in os.listdir(ds_folder):
#     if 


low_exp_img = 'lei'
high_exp_img = 'hei'

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


class Dataset(torch.utils.data.Dataset):
    def __init__(self, text_path, main_dir='Sony/', res=(512,512), tp='bayer', transforms=None):
        self.main_dir = main_dir
        self.Xs = []
        with open(text_path) as f:
            lines = f.readlines()
            for line in lines:
                idx = line.find(' ')
                # self.Xs[line[:idx]] = line[idx+1:]
                self.Xs.append((line[:idx], line[idx+1:]))
        self.transforms = transforms
        self.ids = np.arange(len(self.Xs))
    def __len__(self):
        return len(self.Xs)
    
    def __getitem__(self, idx):
        if idx == 0: 
            self.ids = np.random.permutation(self.ids)
        # return torch.rand(1024), torch.rand(1024)
        x_img_info, gt_img_info = self.Xs[self.ids[idx]]
        x_img_path = os.path.join(self.main_dir, x_img_info)
        gt_img_path = gt_img_info.split(' ')[0]
        gt_img_path = os.path.join(self.main_dir, gt_img_path)
        # print(x_img_path, gt_img_path)
        base_x, base_gt = x_img_path.split('/')[-1], gt_img_path.split('/')[-1]
        in_exposure = float(base_x[9:-5])
        gt_exposure = float(base_gt[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)
        x, gt = rawpy.imread(x_img_path), rawpy.imread(gt_img_path)
        x = np.expand_dims(pack_raw(x), axis=0) * ratio
        
        gt = gt.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt = np.expand_dims(np.float32(gt / 65535.0), axis=0)
        sample = {low_exp_img: x, high_exp_img: gt}
        if self.transforms != None:
            sample = self.transforms(sample)
        return sample

class Dataset2(torch.utils.data.Dataset):
    def __init__(self, text_path, main_dir='Sony/', res=(512,512), tp='bayer', transforms=None):
        self.main_dir = main_dir
        self.Xs = []
        self.X_images = {}
        self.GT_images = {}
        with open(text_path) as f:
            lines = f.readlines()
            for line in lines:
                idx = line.find(' ')
                # self.Xs[line[:idx]] = line[idx+1:]
                self.Xs.append((line[:idx], line[idx+1:]))
        self.transforms = transforms
        self.ids = np.arange(len(self.Xs))
    def __len__(self):
        return len(self.Xs)
    
    def __getitem__(self, idx):
        x_img_info, gt_img_info = self.Xs[self.ids[idx]]
        x_img_path = os.path.join(self.main_dir, x_img_info)
        gt_img_path = gt_img_info.split(' ')[0]
        gt_img_path = os.path.join(self.main_dir, gt_img_path)
        x, gt = None, None 
        if x_img_path in self.X_images.keys():
            x = self.X_images[x_img_path]
        else:
            base_x, base_gt = x_img_path.split('/')[-1], gt_img_path.split('/')[-1]
            in_exposure = float(base_x[9:-5])
            gt_exposure = float(base_gt[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)
            x = rawpy.imread(x_img_path)
            x = np.expand_dims(pack_raw(x), axis=0) * ratio
            self.X_images[x_img_path] = x
        if gt_img_path in self.GT_images.keys():
            gt = self.GT_images[gt_img_path]
        else: 
            gt = rawpy.imread(gt_img_path)
            gt = gt.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt = np.expand_dims(np.float32(gt / 65535.0), axis=0)
            self.GT_images[gt_img_path] = gt
            
        sample = {low_exp_img: x, high_exp_img: gt}
        if self.transforms != None:
            sample = self.transforms(sample)
        return sample
        
lki = None
class RandomCrop(object):
    def __init__(self, output_size) -> None:
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.h, self.w = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.h, self.w = output_size
    def __call__(self, sample):
        global lki
        lli, hli = sample[low_exp_img], sample[high_exp_img]
        lki = lli
        # print(lli.shape, hli.shape)
        or_h, or_w = lli.shape[1:3]
        # print('height width: ', or_h, or_w)
        start_height = torch.randint(0, or_h-self.h, size=(1,))
        start_width = torch.randint(0, or_w-self.w, size=(1,))
        # print(start_height, start_width)
        lli = lli[:, start_height:start_height+self.h, start_width:start_width+self.w]
        hli = hli[:, start_height*2:start_height*2+self.h*2, start_width*2:start_width*2+self.w*2]
        # print(lli.shape, hli.shape)
        return {low_exp_img: lli, high_exp_img:hli}

class RandomFlip(object):
    def __init__(self, probabilty=.3):
        self.probabilty = probabilty
    def __call__(self, sample):
        lli, hli = sample[low_exp_img], sample[high_exp_img]
        hor_prob = torch.rand(1)[0]
        ver_prob = torch.rand(1)[0]
        if hor_prob > self.probabilty:
            lli = np.flip(lli, axis=1)
            hli = np.flip(hli, axis=1)
        if ver_prob > self.probabilty:
            lli = np.flip(lli, axis=2)
            hli = np.flip(hli, axis=2)
        return {low_exp_img: lli, high_exp_img:hli}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        lli, hli = sample[low_exp_img].copy(), sample[high_exp_img].copy()
        # print('coming_shape: ', lli.shape, hli.shape)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        lli = lli.transpose((0, 3, 1, 2))
        hli = hli.transpose((0, 3, 1, 2))
        lli = lli[0]
        hhi = hli[0]
        return {low_exp_img: torch.from_numpy(lli),
                high_exp_img: torch.from_numpy(hli)}


batch_size = 8
workers = 8
im_size = (512, 512)

composer = transforms.Compose([
    RandomCrop(im_size),
    # torchvision.transforms.RandomEqualize(.2),
    RandomFlip(.4),
    # torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.RandomVerticalFlip(),
    ToTensor(),
])



import math
class OriginalModel(torch.nn.Module):
    def __init__(self, block_size, in_channel=4, kernel_size=3, dialation=1) -> None:
        super().__init__()
        self.block_size = block_size 
        
        self.activation = torch.nn.LeakyReLU(.02, True)
        self.max_pool = torch.nn.MaxPool2d(2, 2, 0, ceil_mode=True)
        
        self.convF1 = torch.nn.Conv2d(in_channel, 32, kernel_size, 1, 1, dialation, bias=True)
        self.convF2 = torch.nn.Conv2d(32, 32, kernel_size, 1, 1, dialation, bias=True)
        
        self.convF3 = torch.nn.Conv2d(32, 64, kernel_size, 1, 1, dialation, bias=True)
        self.convF4 = torch.nn.Conv2d(64, 64, kernel_size, 1, 1, dialation, bias=True)
        
        self.convF5 = torch.nn.Conv2d(64, 128, kernel_size, 1, 1, dialation, bias=True)
        self.convF6 = torch.nn.Conv2d(128, 128, kernel_size, 1, 1, dialation, bias=True)
        
        self.convF7 = torch.nn.Conv2d(128, 256, kernel_size, 1, 1, dialation, bias=True)
        self.convF8 = torch.nn.Conv2d(256, 256, kernel_size, 1, 1, dialation, bias=True)
        
        self.convF9 = torch.nn.Conv2d(256, 512, kernel_size, 1, 1, dialation, bias=True)
        self.convF10 = torch.nn.Conv2d(512, 512, kernel_size, 1, 1, dialation, bias=True)
        
        self.conv_upB10 = torch.nn.ConvTranspose2d(512, 256, 2, 2, bias=False)
        self.convB10 = torch.nn.Conv2d(512, 256, kernel_size, 1, 1, dialation, bias=True)
        self.convB9 = torch.nn.Conv2d(256, 256, kernel_size, 1, 1, dialation, bias=True)
        
        self.conv_upB8 = torch.nn.ConvTranspose2d(256, 128, 2, 2, bias=False)
        self.convB8 = torch.nn.Conv2d(256, 128, kernel_size, 1, 1, dialation, bias=True)
        self.convB7 = torch.nn.Conv2d(128, 128, kernel_size, 1, 1, dialation, bias=True)
        
        self.conv_upB6 = torch.nn.ConvTranspose2d(128, 64, 2, 2, bias=False)
        self.convB6 = torch.nn.Conv2d(128, 64, kernel_size, 1, 1, dialation, bias=True)
        self.convB5 = torch.nn.Conv2d(64, 64, kernel_size, 1, 1, dialation, bias=True)
        
        self.conv_upB4 = torch.nn.ConvTranspose2d(64, 32, 2, 2, bias=False)
        self.convB4 = torch.nn.Conv2d(64, 32, kernel_size, 1, 1, dialation, bias=True)
        self.convB3 = torch.nn.Conv2d(32, 32, kernel_size, 1, 1, dialation, bias=True)
        
        self.convB = torch.nn.Conv2d(32, 3 * self.block_size * self.block_size, 1, 1, 0, bias=True)
        
    def init_weights(self, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
    def forward(self, x):
        x = self.convF1(x)
        x = self.activation(x)
        x = self.convF2(x)
        x = self.activation(x)
        up2 = x
        x = self.max_pool(x)
        
        x = self.convF3(x)
        x = self.activation(x)
        x = self.convF4(x)
        x = self.activation(x)
        up4 = x
        x = self.max_pool(x)
        
        x = self.convF5(x)
        x = self.activation(x)
        x = self.convF6(x)
        x = self.activation(x)
        up6 = x
        x = self.max_pool(x)
        
        x = self.convF7(x)
        x = self.activation(x)
        x = self.convF8(x)
        x = self.activation(x)
        up8 = x
        x = self.max_pool(x)
        
        x = self.convF9(x)
        x = self.activation(x)
        x = self.convF10(x)
        x = self.activation(x)
        
        x = self.conv_upB10(x)
        x = torch.cat((x[:, :, :up8.size(2), :up8.size(3)], up8), 1)
        x = self.convB10(x)
        x = self.activation(x)
        x = self.convB9(x)
        x = self.activation(x)
        
        x = self.conv_upB8(x)
        x = torch.cat((x[:, :, :up6.size(2), :up6.size(3)], up6), 1)
        x = self.convB8(x)
        x = self.activation(x)
        x = self.convB7(x)
        x = self.activation(x)
        
        x = self.conv_upB6(x)
        x = torch.cat((x[:, :, :up4.size(2), :up4.size(3)], up4), 1)
        x = self.convB6(x)
        x = self.activation(x)
        x = self.convB5(x)
        x = self.activation(x)
        
        
        x = self.conv_upB4(x)
        x = torch.cat((x[:, :, :up2.size(2), :up2.size(3)], up2), 1)
        x = self.convB4(x)
        x = self.activation(x)
        x = self.convB3(x)
        x = self.activation(x)
        
        x = self.convB(x)
        x = torch.nn.PixelShuffle(2)(x)
        return x
        
    def pixel_shuffle(x, upscale_factor, depth_first=False):
        pass
    
if __name__ == '__main__':
    print('starting main')
    ds = Dataset('./Sony/Sony_train_list.txt', res=im_size, transforms=composer)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    val_ds = Dataset2('./Sony/Sony_val_list.txt', res=im_size, transforms=composer)
    val_dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    model = OriginalModel(2, 4, 3).to(device=cuda)
    model.load_state_dict(torch.load('./lsid__16649091_5.043622644705714'))
    # model.load_state_dict()
    optim = torch.optim.NAdam(model.parameters(),lr=.0003, weight_decay=4e-3)
    mse = torch.nn.MSELoss()
    l1 = torch.nn.L1Loss().to(cuda)
    # l2 = torch.nn
    lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', .5, patience=10)
    # lr = torch.optim.lr_scheduler.ExponentialLR(optim, .95, 30)
    epochs = 50
    print_every = 250 # steps
    iters = 0
    running_loss = 0.0
    model.zero_grad()
    for epoch in range(epochs): 
        model.train()
        for idx, batched in enumerate(dl, 0):
            optim.zero_grad()
            llis, hlis = batched[low_exp_img], batched[high_exp_img]
            outputs = model(llis.to(device=cuda))
            outputs = outputs.to('cpu')
            # hlis.to(cuda)
            err = l1(hlis, outputs)
            err.backward()
            optim.step()
            running_loss += err.item()
            iters+=1
            if (iters % print_every == 0) or ((epoch == epochs-1) and (idx == len(dl)-1)):
                print('[%d/%d][%d/%d]\tloss::%f\trunning_loss::%f' % (epoch, epochs, idx, len(dl), err, running_loss / ((idx+1))))
        print(f'Epoch {epoch+1} \t\t Training Loss: {running_loss / len(dl)}')
        model.eval()
        val_loss = 0.0
        for idx, batched in enumerate(val_dl, 0):
            llis, hlis = batched[low_exp_img], batched[high_exp_img]
            outputs = model(llis.to(device=cuda))
            outputs = outputs.to('cpu')
            err = l1(hlis, outputs)
            val_loss += err.item()
        print(f'Epoch {epoch+1} \t\t Validation Loss: {val_loss / len(val_dl)}')
            
        torch.save(model.state_dict(), './lsid_'+str(epoch)+'_'+str(time.time())[:-10]+'_'+str(val_loss / len(val_dl)))
        running_loss = 0.0
        lr.step(running_loss, epoch)
        # lr.step()