import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import cv2
import numpy as np

def get_paths_images(dir):
    images = []
    paths = []

    for root, _, fnames in os.walk(dir):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            paths.append(path)
            images.append(fname)

    return images, paths


class data_map(Dataset):
    def __init__(self,  dataroot, mask_location ):

        self.dataroot_gars = os.path.join(dataroot, "gars")
        self.dataroot_coords = os.path.join(dataroot, "cords")
        self.dataroot_uv = os.path.join(dataroot, "uvs")
        self.dataroot_mask = os.path.join(dataroot, "masks")

        self.size = 256

        self.images, self.paths = get_paths_images(self.dataroot_uv)
        self.data_size = len(self.images)

        self.transforms = transforms.Compose(
            [   transforms.Resize((self.size, self.size)),
                transforms.ToTensor()   ]
        )

        self.tnsfm_mask = transforms.Compose(
            [   transforms.Resize((self.size, self.size), interpolation=0),
                transforms.ToTensor()   ]
        )

        self.transforms_cords = transforms.Compose(
            [   transforms.ToTensor()   ]
        )
        self.uv_mask = self.tnsfm_mask(Image.open(mask_location))
        self.uv_mask = torch.cat((self.uv_mask, self.uv_mask), dim = 0 )


    def get_img_rep(self, mask):
        # grid2 = np.meshgrid(np.linspace(-1,1,256), np.linspace(-1,1,256))

        x = torch.from_numpy(np.linspace(-1, 1, self.size))
        y = torch.from_numpy(np.linspace(-1, 1, self.size))

        xx = x.view(-1, 1).repeat(1, self.size)
        # print(xx)
        yy = y.repeat(self.size, 1)
        # print(yy)
        meshed = torch.cat([yy.unsqueeze_(2), xx.unsqueeze_(2)], 2)
        meshed = meshed.permute(2,0,1)
        # print (meshed.size())

        # print(mask.size())
        return mask[0:2, :, :].float() * meshed.float()

    def __getitem__(self, index):

        path_mask = os.path.join(self.dataroot_mask, self.images[index])

        mask = Image.open(path_mask)
        mask = np.array(mask)
        mask = mask.astype(np.float32)
        mask = mask / 255
        mask = np.round(mask)
        mask = Image.fromarray(np.uint8(mask) * 255)
        mask = self.tnsfm_mask(mask)
        mask = torch.cat((mask, mask, mask), 0)

        path_gar = os.path.join(self.dataroot_gars, self.images[index])
        gar = self.transforms(Image.open(path_gar))

        gar =  mask * gar
        gar_rep = self.get_img_rep( mask)

        temp_name = self.images[index]
        temp_name = temp_name.replace(".jpg", ".npy")

        path_cords = os.path.join(self.dataroot_coords, temp_name)
        path_uv = os.path.join(self.dataroot_uv, self.images[index])

        cords = np.load(path_cords)
        cords = cv2.resize(cords, (self.size, self.size))

        cords = self.transforms_cords(cords)
        cords = self.uv_mask.float() * cords.float()

        uv = self.transforms(Image.open(path_uv))
        uv = uv.float()

        return {'gar': gar, 'gar_rep': gar_rep, 'cords': cords, 'uv': uv, 'uv_mask': self.uv_mask}

    def __len__(self):
        return self.data_size



class data_seg(Dataset):
    def __init__(self, dataroot):
        self.size = 256
        self.dataroot_gar = os.path.join(dataroot, "gars")
        self.dataroot_mask = os.path.join(dataroot, "masks")
        self.images, self.paths = get_paths_images(self.dataroot_mask)

        self.total_size = len(self.images)

        self.transform_img = transforms.Compose(
            [   transforms.Resize((self.size, self.size)),
                transforms.ColorJitter(hue=.5, saturation=.5, contrast = 0.7),
                transforms.ToTensor()         ]
        )
        self.transform_mask = transforms.Compose(
            [   transforms.Resize((self.size, self.size), interpolation = 0),
                transforms.ToTensor()      ]
            )

    def get_gt(self, image):
        if self.use_soft:
            sil = torch.zeros(image.size(), dtype=torch.long)
        else:
            sil = torch.zeros(image.size())

        sil[image == 1] = 1  # background = 1 and foreground = 0
        if self.use_soft:
            sil = sil.squeeze(0)
        return sil

    def __getitem__(self, index):

        path_mask = self.paths[index]
        mask = Image.open(path_mask)
        mask = np.array(mask)
        mask = mask.astype(np.float32)
        mask = mask / 255
        mask = np.round(mask)

        mask = Image.fromarray(np.uint8(mask) * 255)

        mask = self.transform_mask(mask)
        mask = mask.squeeze(0)
        mask = mask.long()

        image_path = os.path.join(self.dataroot_gar, self.images[index])
        image = self.transform_img(Image.open(image_path).convert("RGB"))

        return {'img': image, 'gt': mask, 'img_name': self.images[index]}

    def __len__(self):
        return self.total_size
