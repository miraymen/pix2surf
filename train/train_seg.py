import argparse
import network
import data
from torch.utils.data import DataLoader
import torch
import time
from tensorboardX import SummaryWriter
import torch.optim as opt
import os
import shutil
import datetime
import cv2
import torch.nn as nn

from utils import *

class Solver():
    def __init__(self):
        self.args = self.get_args()

        self.model_name = str(datetime.date.today() )+ '_seg_net_'+self.args.gar_type
        self.get_gpus()
        if self.args.gar_type == 'shirts':
            if self.args.side:
                self.args.dataroot = os.path.join(self.args.dataroot, 'front')
                self.model_name = self.model_name + '_front'
            else:
                self.args.dataroot = self.args.dataroot + 'back/'
                self.model_name = self.model_name + '_back'

        self.dirs = {'model_dir':"./saved_models",
                     'image_dir': "./saved_images",
                     'log_dir':"./saved_logs"}

        for key, val in self.dirs.items():
            self.dirs[key] = os.path.join(val, self.model_name)
            self.make_dirs(self.dirs[key])

        self.train_dataset = data.data_seg(dataroot=self.args.dataroot)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size,
                                           num_workers=1, shuffle=True)

        self.iter_p_batch = len(self.train_dataset) // self.args.batch_size

        self.network = network.UnetGenerator(input_nc=3, output_nc=2, num_downs=8, ngf=64, norm_layer=nn.InstanceNorm2d)
        self.print_network(self.network)
        self.network.to(self.device)

        self.writer = SummaryWriter(self.dirs['log_dir'])

        self.optim = opt.Adam(params=self.network.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

        self.criterion = torch.nn.CrossEntropyLoss()

    def make_dirs(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            shutil.rmtree(dir)
            os.makedirs(dir)

    def get_gpus(self):
        gpus = []
        for s in list(self.args.gpu_ids):
            if (s.isdigit()):
                gpus.append(int(s))

        if gpus[0] == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda", index=gpus[0])

        self.args.gpu_ids = gpus

    def get_args(self):
        ap = argparse.ArgumentParser()

        ap.add_argument("-b", "--batch_size", type=int, default=1, help="input batch size")
        ap.add_argument("-e", "--num_epochs", type=int, default=20, help="number of epochs")
        ap.add_argument("-g", "--gpu_ids", type=str, default="0", help="gpu ids, -1 for cpu")
        ap.add_argument("-dr", "--dataroot", type=str, default="./data/shirts", help="location of training datset")
        ap.add_argument("-w", "--num_workers", type=int, default=1, help="number of workers")
        ap.add_argument("-lr", "--lr", type=float, default=0.0001, help="Learning rate")
        ap.add_argument("-c", "--check_print", type=int, default=100, help="check point after these iterations")
        ap.add_argument("-s", "--save_point", type=int, default=10, help="Save models after this epoch")
        ap.add_argument("--side", type = str2bool, default = True, help = "Whether to use the fron or the back")
        ap.add_argument("--gar_type", type = str, default = 'shirts', help = 'shirts|pants|shorts')
        args = ap.parse_args()
        return args

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            if p.requires_grad == True:
                num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def train(self):

        start_time = time.time()
        for epoch in range(self.args.num_epochs):
            print('\nModel {}. Epoch : {}'.format(self.model_name, epoch + 1))
            loss = {}
            for step, sample in enumerate(self.train_dataloader):

                img = sample['img'].to(self.device)
                gt = sample['gt'].to(self.device)

                self.optim.zero_grad()
                out = self.network(img)
                loss_curr = self.criterion(out, gt)
                loss_curr.backward()
                self.optim.step()
                loss["loss"] = loss_curr.item()

                if (step + 1) % self.args.check_print == 0:
                    global_iter = (epoch) * self.iter_p_batch + step + 1
                    for tag, value in loss.items():
                        self.writer.add_scalar(tag, value, global_iter)

                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Epoch [{}] Iteration [{}/{}]".format(et, epoch + 1, step + 1, self.iter_p_batch)

                    print(log)
                    img = img[0,:,:, :]
                    gt = gt[0, :,:]

                    m = torch.nn.Softmax2d()
                    out = m(out)
                    out = out.squeeze(0)

                    out_fg = out[1, :, :]
                    out_bg = out[0, :, :]
                    # print(out_fg)
                    out_fg_binary = binarizeimage(out_fg)
                    out_bg_binary = binarizeimage(out_bg)

                    save1 = sv_dsk(img, gt, out_fg_binary, out_bg_binary)
                    save_file = os.path.join(self.dirs['image_dir'], 'epoch{}_iter{}of{}_gliter{}.jpg'.format(epoch + 1,
                                                                                                              step + 1, self.iter_p_batch, global_iter))
                    cv2.imwrite(save_file, save1)

            if ((epoch + 1) % self.args.save_point == 0):
                gen_file = os.path.join(self.dirs['model_dir'],  'network_epoch{}.pt'.format(epoch + 1))
                print('Saving file: ' + gen_file)
                torch.save(self.network.state_dict(), gen_file)

if __name__ == '__main__':
    solve = Solver()
    solve.train()
