import network
import PercepLoss

import torch.nn.functional as F
import argparse

import torch.nn as nn
from abc import ABC
import datetime

from collections import OrderedDict

import os
import data
from torch.utils.data import DataLoader
import torch
import time
from tensorboardX import SummaryWriter
import cv2

import shutil

from utils import *

class MapNet(ABC):
    """
    This class implements the mapping network in Pix2Surf
    """
    def __init__(self):
        """
        Initialize the MapNet class
        """
        self.get_args()
        self.get_gpus()
        from datetime import datetime
        time_dt = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.model_name = time_dt + '_mask_net_'+self.args.gar_type
        TWO_UP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if self.args.side:
            self.args.dataroot = os.path.join('./data/{}/front'.format(self.args.gar_type))
            self.model_name = self.model_name + '_front'
            self.args.mask_location  = os.path.join(TWO_UP, 'assets/uv_masks/{}_front.jpg'.format(self.args.gar_type))
        else:
            self.args.dataroot = os.path.join('./data/{}/back'.format(self.args.gar_type))
            self.model_name = self.model_name + '_back'
            self.args.mask_location  = os.path.join(TWO_UP, 'assets/uv_masks/{}_back.jpg'.format(self.args.gar_type))


        self.dirs = {'model_dir':"./saved_models",
                     'image_dir': "./saved_images",
                     'log_dir':"./saved_logs"}

        for key, val in self.dirs.items():
            self.dirs[key] = os.path.join(val, self.model_name)
            self.make_dirs(self.dirs[key])

        self.losses = {}
        self.visual_names = ['gar', 'uv', 'uv_out']

        self.net_map = network.ResnetGenerator(2, 2, 0, 64, n_blocks=6, norm_layer = nn.InstanceNorm2d)

        self.print_network(self.net_map)

        self.net_map.to(self.device)

        self.criterion_reg = torch.nn.MSELoss()
        self.criterion_L1 = torch.nn.L1Loss()
        self.criterion_percept = PercepLoss.VGGLoss(device=self.device, use_perceptual=True, imagenet_norm=True)

        self.optimizer_G = torch.optim.Adam(self.net_map.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

        self.train_dataset = data.data_map(dataroot=self.args.dataroot, mask_location = self.args.mask_location)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=1, num_workers=self.args.num_workers, shuffle=True)
        self.args.iter_p_batch = len(self.train_dataset) // self.args.batch_size

        self.writer = SummaryWriter(self.dirs['log_dir'])

    def get_args(self):
        """ Builds argparse and accepts arguments """
        ap = argparse.ArgumentParser()
        ap.add_argument("-b", "--batch_size", type=int, default=1, help="input batch size")
        ap.add_argument("-e", "--num_epochs", type=int, default=200, help="number of epochs")
        ap.add_argument("-g", "--gpu_ids", type=str, default="-1", help="gpu ids, -1 for cpu")

        ap.add_argument("-w", "--num_workers", type=int, default=1, help="number of workers")
        ap.add_argument("-lr", "--lr", type=float, default=0.0001, help="Learning rate")
        ap.add_argument("-c", "--check", type=int, default=100, help="check point after these iterations")
        ap.add_argument("-s", "--save_point", type=int, default=50, help="Save models after this epoch")

        ap.add_argument("--lambda_reg", type=float, default=1, help="Regresion loss")
        ap.add_argument("--lambda_recon", type=float, default=0, help="Reconstruction loss at the end")
        ap.add_argument("--lambda_per", type=float, default=0, help="Weight of perceptual loss")

        ap.add_argument("--gar_type", type = str, default = "shirts", help = "shirts|pants|shorts")
        ap.add_argument("--side", type=str2bool, default=True, help="Whether the front or the back is being processed")
        args = ap.parse_args()

        self.args = args

    def print_network(self, model):
        """Prints the input network"""
        num_params = 0
        for p in model.parameters():
            if p.requires_grad == True:
                num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def make_dirs(self, dir):
        """Make directories for storing images, logs and models"""
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            shutil.rmtree(dir)
            os.makedirs(dir)

    def get_gpus(self):
        """Add device on which the code will run"""
        gpus = []
        for s in list(self.args.gpu_ids):
            if (s.isdigit()):
                gpus.append(int(s))
        if gpus[0] == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda", index=gpus[0])

        self.args.gpu_ids = gpus

    def get_current_visuals(self):
        """Return visualization images"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)[0, :, :, :]
        return visual_ret

    def set_input(self, sample):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            sample (dict): The current sample from the dataloader.
        """
        self.gar = sample['gar'].to(self.device)
        self.gar_rep = sample['gar_rep'].to(self.device)
        self.cords = sample['cords'].to(self.device)
        self.uv = sample['uv'].to(self.device)
        self.uv_mask = sample['uv_mask'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        out = self.net_map(self.gar_rep)  # synth2real(synth)
        out = out.permute(0, 2, 3, 1)
        self.uv_out = F.grid_sample(self.gar, out)
        self.out = out.permute(0, 3, 1, 2)


    def backward_G(self):
        """Calculate the loss for mapping network"""

        loss_total = 0

        if self.args.lambda_reg:
            self.loss_reg = self.criterion_reg(self.uv_mask * self.out, self.uv_mask * self.cords)
            self.losses["G/reg"] = self.loss_reg.item()

            loss_total = loss_total + self.args.lambda_reg * self.loss_reg

        if self.args.lambda_recon:
            # print("recon")
            self.loss_recon = self.criterion_L1(self.uv_out, self.uv)
            self.losses["G/recon"] = self.loss_recon.item()

            loss_total = loss_total + self.args.lambda_recon * self.loss_recon

        if self.args.lambda_per:
            # print("recon")
            self.loss_per = self.criterion_percept(self.uv_out, self.uv)
            self.losses["G/per"] = self.loss_per.item()

            loss_total = loss_total + self.args.lambda_per * self.loss_per

        self.loss_G = loss_total

        self.loss_G.backward()

        self.losses["G/total"] = self.loss_G.item()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        # Optimize the network
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def train(self):
        """Train the mapping network. Called just one"""

        self.start_time = time.time()
        for epoch in range(self.args.num_epochs):
            print('\nModel {}. Epoch : {}'.format(self.model_name, epoch + 1))

            for step, sample in enumerate(self.train_dataloader):
                self.set_input(sample)
                self.optimize_parameters()

                if (step + 1) % self.args.check == 0:
                    # Print out training information.
                    et = time.time() - self.start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Epoch [{}] Iteration [{}/{}]".format(et, epoch + 1, step , self.args.iter_p_batch)
                    print(log)
                    visuals = self.get_current_visuals()
                    global_iter = (epoch) * self.args.iter_p_batch + step
                    save_img = sv_dsk_map(visuals)

                    save_file = os.path.join(self.dirs['image_dir'], 'epoch{}_iter{}of{}_gliter{}.jpg'.format(epoch + 1, step + 1, self.args.iter_p_batch,
                                                                                      global_iter))
                    cv2.imwrite(save_file, save_img)

                    for tag, value in self.losses.items():
                        self.writer.add_scalar(tag, value, global_iter)

            if ((epoch + 1) % self.args.save_point == 0):
                gen_file = os.path.join(self.dirs['model_dir'], 'network_epoch{}.pt'.format(epoch + 1))
                print('Saving file: ' + gen_file)
                torch.save(self.net_map.state_dict(), gen_file)

if __name__ == "__main__":
    solver = MapNet()
    solver.train()
