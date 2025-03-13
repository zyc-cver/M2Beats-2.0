import numpy as np
import torch
import torch.cuda
from torch import nn
from module.st_gcn.st_gcn_full_multi_branch import ST_GCN_18
import os

class m2bnet(nn.Module):
    def __init__(self, d_model=64, is_training=True):
        super(m2bnet, self).__init__()
        self.d_model = d_model

        graph_cfg = {
            "layout": 'coco',
            "strategy": "spatial"
        }
        self.pose_net = ST_GCN_18(3, d_model*4, 10, graph_cfg, dropout=0.1)

        self.linear1 = nn.Linear(d_model,256)
        self.linear2 = nn.Linear(256,d_model)
        self.relu = nn.ReLU()

        self.proj_beat = nn.Linear(d_model, 1)
        self.sig=nn.Sigmoid()
        self.patch_size = 16
        self.vit_linear=nn.Linear(256,d_model)
    
    def forward_output(self, h):
        y_beat = self.proj_beat(h)
        y_beat=self.sig(y_beat)
        return y_beat
    
    def forward_pose_net(self, pose):
        N, C, T, V, M = pose.size()
        pose = self.pose_net(pose)  # output: (batch, d_model*4, frame/4)
        pose = pose.permute(0,2,1)  # (batch, frame/4, d_model*4)
        pose = pose.reshape(N, T, self.d_model)  # (batch, frame, d_model)

        return pose
    
    def forward(self, x):   
        N, T, V, C = x.size()
        x = x.permute(0,3,1,2)
        x = x.view((N,C,T,V,1))  
        x = self.forward_pose_net(x)

        h = self.linear1(x)
        h = self.relu(h)
        h = self.linear2(h)
        h = self.relu(h)
        y_beat = self.forward_output(h)
        return y_beat
    
    def inference(self, x):
        h = self.forward_hidden(x)
        y_beat= self.forward_output(h)

        y_beat = y_beat[:, ...].permute(0, 2, 1)

        return y_beat