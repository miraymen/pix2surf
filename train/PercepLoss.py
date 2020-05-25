## Implementing perceptual loss using VGG19
##-------------------------------------------------------
import torch
import torch.nn as nn
import os
from torchvision import models

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False, only_last = False, final_feat_size=8):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.only_last = only_last
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            if final_feat_size <=64 or type(vgg_pretrained_features[x]) is not nn.MaxPool2d:
                self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            if final_feat_size <=32 or type(vgg_pretrained_features[x]) is not nn.MaxPool2d:
                self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            if final_feat_size <=16 or type(vgg_pretrained_features[x]) is not nn.MaxPool2d:
                self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            if final_feat_size <=8 or type(vgg_pretrained_features[x]) is not nn.MaxPool2d:
                self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        if self.only_last:
            return h_relu5
        else:
            out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
            return out

###--------Probably start here --------------

class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

def normalize_tensor(in_feat,eps=1e-10):
    # norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1)).view(in_feat.size()[0],1,in_feat.size()[2],in_feat.size()[3]).repeat(1,in_feat.size()[1],1,1)
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1)).view(in_feat.size()[0],1,in_feat.size()[2],in_feat.size()[3])

    return in_feat/(norm_factor.expand_as(in_feat)+eps)

class squeezenet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = models.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2,5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])


        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        out = [h_relu1,h_relu2,h_relu3,h_relu4,h_relu5,h_relu6,h_relu7]

        return out


class VGGLoss(nn.Module):
    def __init__(self,  device, network = 'vgg', use_perceptual=True, imagenet_norm = False, use_style_loss=0):
        super(VGGLoss, self).__init__()
        self.criterion = nn.L1Loss()

        self.use_style_loss = use_style_loss

        if network == 'vgg':
            self.chns = [64,128,256,512,512]
        else:
            self.chns = [64,128,256,384,384,512,512]

        if use_perceptual:
            self.use_perceptual = True
            self.lin0 = NetLinLayer(self.chns[0],use_dropout=False)
            self.lin1 = NetLinLayer(self.chns[1],use_dropout=False)
            self.lin2 = NetLinLayer(self.chns[2],use_dropout=False)
            self.lin3 = NetLinLayer(self.chns[3],use_dropout=False)
            self.lin4 = NetLinLayer(self.chns[4],use_dropout=False)
            self.lin0.to(device)
            self.lin1.to(device)
            self.lin2.to(device)
            self.lin3.to(device)
            self.lin4.to(device)


        self.imagenet_norm = imagenet_norm
        if not self.imagenet_norm:
            self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1)).to(device)
            self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1)).to(device)

        self.net_type = network
        if network == 'vgg':
            self.pnet = Vgg19().to(device)
            self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        else:
            self.pnet = squeezenet().to(device)
            self.weights = [1.0]*7
            if use_perceptual:
                self.lin5 = NetLinLayer(self.chns[5],use_dropout=False)
                self.lin6 = NetLinLayer(self.chns[6],use_dropout=False)
                self.lin5.to(device)
                self.lin6.to(device)
        if self.use_perceptual:
            my_path = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(my_path, "../pretrained/")
            self.load_state_dict(torch.load(os.path.join(path, network+'.pth')), strict=False)
        for param in self.parameters():
            param.requires_grad = False

    def gram(self, x):
        a, b, c, d = x.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = x.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def gram2(self, x):
        a, b, c, d = x.size()  # a=batch size
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        G = torch.zeros(a, b, b)

        for j in range(a):
            features = x[j, :, :, :].view(b, c * d)  # resise F_XL into \hat F_XL
            G[j, :, :] = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(b * c * d)

    def gram3(self, x):
        a, b, c, d = x.size()  # a=batch size
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        features = x.view(a, b, c*d)

        G = torch.bmm(features, torch.transpose(features, 1, 2))

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(b * c * d)


    def forward(self, x, y):

        x, y = x.expand(x.size(0), 3, x.size(2), x.size(3)), y.expand(y.size(0), 3, y.size(2), y.size(3))

        if not self.imagenet_norm:
            x = (x - self.shift.expand_as(x))/self.scale.expand_as(x)
            y = (y - self.shift.expand_as(y))/self.scale.expand_as(y)

        x_vgg, y_vgg = self.pnet(x), self.pnet(y)
        loss = 0
        if self.use_perceptual:
            normed_x = [normalize_tensor(x_vgg[kk]) for (kk, out0) in enumerate(x_vgg)]
            normed_y = [normalize_tensor(y_vgg[kk]) for (kk, out0) in enumerate(y_vgg)]
            diffs = [(normed_x[kk]-normed_y[kk].detach())**2 for (kk,out0) in enumerate(x_vgg)]
            loss = self.lin0.model(diffs[0]).mean()
            loss = loss + self.lin1.model(diffs[1]).mean()
            loss = loss + self.lin2.model(diffs[2]).mean()
            loss = loss + self.lin3.model(diffs[3]).mean()
            loss = loss + self.lin4.model(diffs[4]).mean()
            if(self.net_type=='squeeze'):
                loss = loss + self.lin5.model(diffs[5]).mean()
                loss = loss + self.lin6.model(diffs[6]).mean()
            if self.use_style_loss:
                style_loss = 0.
                for kk in range(3, len(x_vgg)):
                    style_loss += self.criterion(self.gram3(x_vgg[kk]), self.gram3(y_vgg[kk]))
                loss += self.use_style_loss * style_loss
        else:
            for i in range(len(x_vgg)):
                loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


