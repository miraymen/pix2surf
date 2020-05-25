import argparse
import torch
import torch.nn as nn
from train import network
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import cv2
import os
import glob

from train.utils import *

class Demo():
    def __init__(self):
        self.get_args()
        self.get_path()
        self.get_gpus()
        self.transform = transforms.Compose(
            [  transforms.Resize((256, 256)),
                transforms.ToTensor() ]  )

        self.net_map = network.ResnetGenerator(2, 2, 0, 64, n_blocks=6, norm_layer = nn.InstanceNorm2d)
        self.net_seg = network.UnetGenerator(input_nc=3, output_nc=2, num_downs=8, ngf=64, norm_layer=nn.InstanceNorm2d)

        if not os.path.isdir(self.opt.output):
            os.makedirs(self.opt.output)

        if not os.path.isdir(self.opt.video):
            os.makedirs(self.opt.video)

    def get_args(self):
        ap = argparse.ArgumentParser()

        ap.add_argument("--gpus", type = str, default = '0', help = "-1 for cpu")
        ap.add_argument("--pose_id", type = str, default = '0')
        ap.add_argument("--img_id", type = str, default = '0')
        ap.add_argument("--low_type", type = str, default = 'shorts', help = 'pants|shorts')

        ap.add_argument("--output", type = str, default = './output', help = "Location where renderings are stored")
        ap.add_argument("--video", type = str, default = './video', help = "location where text maps and videos are stored")

        ap.add_argument("--body_tex", type = str, default = './test_data/images/body_tex/body_tex.jpg')

        self.opt = ap.parse_args()
        assert (int(self.opt.img_id) < 5 and int(self.opt.img_id) >= 0), 'Please enter an img_id between 0 and 4'
        assert (int(self.opt.pose_id) < 5 and int(self.opt.pose_id) >= 0), 'Please enter a pose_id between 0 and 4'
    def get_path(self):
        self.opt.low_mesh = './test_data/meshes/'+ self.opt.low_type + '/lower_{}.obj'.format(self.opt.pose_id)
        self.opt.up_mesh = './test_data/meshes/' + self.opt.low_type + '/upper_{}.obj'.format(self.opt.pose_id)
        self.opt.body_mesh = './test_data/meshes/' + self.opt.low_type + '/body_{}.obj'.format(self.opt.pose_id)


        self.opt.img_up_front = './test_data/images/' + self.opt.low_type + '/shirt{}.jpg'.format(self.opt.img_id)
        self.opt.img_up_back = './test_data/images/' + self.opt.low_type + '/shirt{}_b.jpg'.format(self.opt.img_id)

        self.opt.img_low_front = './test_data/images/' + self.opt.low_type + '/{}{}.jpg'.format(self.opt.low_type, self.opt.img_id)
        self.opt.img_low_back = './test_data/images/' + self.opt.low_type + '/{}{}_b.jpg'.format(self.opt.low_type, self.opt.img_id)

        self.opt.seg_up_front = './pretrained/seg_shirts_front.pt'
        self.opt.seg_up_back = './pretrained/seg_shirts_back.pt'

        self.opt.map_up_front = './pretrained/map_shirts_front.pt'
        self.opt.map_up_back = './pretrained/map_shirts_back.pt'

        self.opt.seg_low_front = './pretrained/seg_{}.pt'.format(self.opt.low_type)
        self.opt.seg_low_back = './pretrained/seg_{}.pt'.format(self.opt.low_type)

        self.opt.map_low_front = './pretrained/map_{}_front.pt'.format(self.opt.low_type)
        self.opt.map_low_back = './pretrained/map_{}_back.pt'.format(self.opt.low_type)

    def get_gpus(self):
        """Add device on which the code will run"""
        gpus = []
        for s in list(self.opt.gpus):
            if (s.isdigit()):
                gpus.append(int(s))
        if gpus[0] == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda", index=gpus[0])

        self.opt.gpu_ids = gpus

    def read_images(self, image_path):
        image = self.transform(Image.open(image_path).convert("RGB"))
        image = image.unsqueeze(0)
        return image.to(self.device)

    def get_img_rep(self, seg_out):
        m = torch.nn.Softmax2d()
        out = m(seg_out)
        out = out.squeeze(0)[1, :, :]
        out_fg_binary = binarizeimage(out)

        x = torch.from_numpy(np.linspace(-1, 1, 256))
        y = torch.from_numpy(np.linspace(-1, 1, 256))

        xx = x.view(-1, 1).repeat(1, 256)
        yy = y.repeat(256, 1)
        meshed = torch.cat([yy.unsqueeze_(2), xx.unsqueeze_(2)], 2)
        meshed = meshed.permute(2, 0, 1)

        out_fg_binary = out_fg_binary.unsqueeze(0)
        mask2 = torch.cat((out_fg_binary, out_fg_binary), dim=0)

        rend_rep = mask2.float() * meshed.float()
        return rend_rep.unsqueeze(0).to(self.device)


    def forward(self):
        dict = ['up_front', 'up_back', 'low_front', 'low_back']
        for val in dict:
            map_net_pth = getattr(self.opt, 'map_'+ val)
            self.net_map.load_state_dict(torch.load(map_net_pth))

            seg_net_pth = getattr(self.opt, 'seg_'+val)
            self.net_seg.load_state_dict(torch.load(seg_net_pth))

            self.net_seg.to(self.device)
            self.net_seg.eval()

            self.net_map.to(self.device)
            self.net_map.eval()

            img_path = getattr(self.opt, 'img_'+val)
            image = self.read_images(img_path)

            output = self.net_seg(image)
            map_in = self.get_img_rep(output)
            map_in = map_in.to(self.device)

            out = self.net_map(map_in)
            out = out.permute(0, 2, 3, 1)
            uv_out = F.grid_sample(image, out)
            setattr(self, 'uv_'+ val, tensor2image(uv_out[0, :, :, :]))

    def combine_textures(self):
        dirs = ['up', 'low']
        for val in dirs:
            cut1 = getattr(self, 'uv_'+val + '_front')
            cut2 = getattr(self, 'uv_' + val + '_back')
            base = np.zeros((2000, 2000, 3))
            cut1 = cv2.resize(cut1, (1000, 1000))
            cut2 = cv2.resize(cut2, (1000, 1000))
            base = base.astype('float64')
            base[500:1500, 0:1000] = cut1
            base[500:1500, 1000:2000] = cut2
            save_file = os.path.join(self.opt.output, val +'.jpg')
            setattr(self.opt, 'tex_loc_' +val, save_file)
            cv2.imwrite(save_file, base)
            setattr(self, 'tex_'+val, base)

    def make_video(self):
        paths = sorted(glob.glob(os.path.join(self.opt.video, '*.png')))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_loc =  './video.mp4'
        video = cv2.VideoWriter(video_loc, fourcc, 15, (175, 350))
        img_fname = paths
        for fname in img_fname:
            img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            trans_mask = img[:, :, 3] == 0
            img[trans_mask] = [255, 255, 255, 255]
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            video.write(img)

        video.release()

    def run(self):
        self.forward()
        self.combine_textures()
        os.system('blender --background --python render.py -- --body_tex {} --body_mesh {} --up_tex {} --up_mesh {} --low_mesh {} --low_tex {} --renderfolder {}'.format(
            self.opt.body_tex, self.opt.body_mesh, self.opt.tex_loc_up, self.opt.up_mesh, self.opt.low_mesh, self.opt.tex_loc_low, self.opt.video
        ))

        self.make_video()
        os.system('rm -r {}'.format(self.opt.output))
        os.system('rm -r {}'.format(self.opt.video))


if __name__ == '__main__':

    demo = Demo()
    demo.run()
