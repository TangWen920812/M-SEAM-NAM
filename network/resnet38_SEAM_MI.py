import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import numpy as np
np.set_printoptions(threshold=np.inf)

import network.resnet38d
from tool import pyutils
import random
import itertools

class Net(network.resnet38d.Net):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(4096, 21, 1, bias=False)
        self.fc8_final = nn.Conv2d(21, 2, 1, bias=False)

        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f8_5 = torch.nn.Conv2d(4096, 256, 1, bias=False) #
        self.f9 = torch.nn.Conv2d(192+3, 192, 1, bias=False)
        self.f10 = torch.nn.Conv2d(448, 448, 1, bias=False) #448, 448
        
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.kaiming_normal_(self.f8_5.weight) #
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)
        torch.nn.init.xavier_uniform_(self.f10.weight, gain=4) #
        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f8_5, self.f9, self.fc8, self.f10] #
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

        self.predefined_featuresize = int(448 // 8)
        self.radius = 5
        self.ind_from, self.ind_to = pyutils.get_indices_of_pairs(radius=self.radius, size=(
        self.predefined_featuresize, self.predefined_featuresize))
        self.ind_from = torch.from_numpy(self.ind_from);
        self.ind_to = torch.from_numpy(self.ind_to)

    def forward(self, x, aff=False):
        N, C, H, W = x.size()
        d = super().forward_as_dict(x)
        cam = self.fc8(self.dropout7(d['conv6']))
        cam = self.fc8_final(cam)
        n,c,h,w = cam.size()
        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n,c,-1), dim=-1)[0].view(n,c,1,1)+1e-5
            cam_d_norm = F.relu(cam_d-1e-5)/cam_d_max
            cam_d_norm[:,0,:,:] = 1-torch.max(cam_d_norm[:,1:,:,:], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:,1:,:,:], dim=1, keepdim=True)[0]
            cam_d_norm[:,1:,:,:][cam_d_norm[:,1:,:,:] < cam_max] = 0

        f8_3 = F.relu(self.f8_3(d['conv4'].detach()), inplace=True)
        f8_4 = F.relu(self.f8_4(d['conv5'].detach()), inplace=True)
        f8_5 = F.relu(self.f8_5(d['conv6'].detach()), inplace=True) #
        x_s = F.interpolate(x,(h,w),mode='bilinear',align_corners=True)
        f = torch.cat([x_s, f8_3, f8_4], dim=1)
        n,c,h,w = f.size()

        cam_rv = F.interpolate(self.PCM(cam_d_norm, f), (H,W), mode='bilinear', align_corners=True)
        cam = F.interpolate(cam, (H,W), mode='bilinear', align_corners=True)

        if aff:
            feature = F.elu(self.f10(torch.cat([f8_3, f8_4, f8_5], dim=1))) # torch.cat([f8_3, f8_4, f8_5], dim=1)
            feature = feature.view(feature.size(0), feature.size(1), -1).contiguous() #
            ind_from = self.ind_from
            ind_to = self.ind_to
            ind_from = ind_from.contiguous()
            ind_to = ind_to.contiguous()
            ff = torch.index_select(feature, dim=2, index=ind_from.cuda(non_blocking=True))
            ft = torch.index_select(feature, dim=2, index=ind_to.cuda(non_blocking=True))
            ff = torch.unsqueeze(ff, dim=2)
            ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))
            aff = torch.exp(-torch.mean(torch.abs(ft - ff), dim=1))
            return cam, cam_rv, aff
        else:
            return cam, cam_rv

    def PCM(self, cam, f):
        n,c,h,w = f.size()
        cam = F.interpolate(cam, (h,w), mode='bilinear', align_corners=True).view(n,-1,h*w)
        f = self.f9(f)
        f = f.view(n,-1,h*w)
        f = f/(torch.norm(f,dim=1,keepdim=True)+1e-5)

        aff = F.relu(torch.matmul(f.transpose(1,2), f),inplace=True)
        aff = aff/(torch.sum(aff,dim=1,keepdim=True)+1e-5)
        cam_rv = torch.matmul(cam, aff).view(n,-1,h,w)
        
        return cam_rv

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


