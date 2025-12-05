# Pedestrian Graph+ with configurable branches: image, velocity, 3D keypoints, segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F


class pedMondel(nn.Module):
    def __init__(self, frames=True, vel=False, seg=False, h3d=True, nodes=19, n_clss=3):
        super(pedMondel, self).__init__()

        self.h3d = h3d
        self.frames = frames
        self.vel = vel
        self.seg = seg
        self.n_clss = n_clss

        self.ch = 4 if h3d else 3
        self.i_ch = 4 if seg else 3

        self.ch1 = 64
        self.ch2 = 128
        self.ch3 = 256

        self.gcn1 = STA_GCN_Unit(self.ch, self.ch1, nodes)
        self.gcn2 = STA_GCN_Unit(self.ch1, self.ch2, nodes)
        self.gcn3 = STA_GCN_Unit(self.ch2, self.ch3, nodes)


        if self.frames:
            self.image_branch = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.i_ch if i == 0 else self.ch1, self.ch1, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(self.ch1, ch),
                    nn.Sigmoid()
                ) for i, ch in enumerate([self.ch1, self.ch2, self.ch3])
            ])

        if self.vel:
            self.velocity_branch = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(2 if i == 0 else self.ch1, self.ch1, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(self.ch1, ch),
                    nn.Sigmoid()
                ) for i, ch in enumerate([self.ch1, self.ch2, self.ch3])
            ])

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # self.att = TransformerVecAttention(self.ch3)

        self.att = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.ch3, self.ch3, bias=False),
            nn.LayerNorm(self.ch3),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.ch3, self.n_clss)

    def forward(self, x_pose, x_image=None, x_velocity=None):
        if not self.h3d:
            x_pose = x_pose[[0, 1, 3], ...]

        if self.frames and not self.seg and x_image is not None:
            x_image = x_image[1:, ...] if x_image.shape[1] == 4 else x_image

        img_feat, vel_feat = x_image, x_velocity

        x = self.gcn1(x_pose)
        img_weight1 = self.image_branch[0](img_feat).unsqueeze(-1).unsqueeze(-1) if self.frames and x_image is not None else 1.0
        vel_weight1 = self.velocity_branch[0](vel_feat).unsqueeze(-1).unsqueeze(-1) if self.vel and x_velocity is not None else 1.0
        x = x * img_weight1 * vel_weight1

        x = self.gcn2(x)
        if self.frames and x_image is not None:
            img_feat = F.relu(self.image_branch[0][0](img_feat))
            img_weight2 = self.image_branch[1](img_feat).unsqueeze(-1).unsqueeze(-1)
        else:
            img_weight2 = 1.0
        if self.vel and x_velocity is not None:
            vel_feat = F.relu(self.velocity_branch[0][0](vel_feat))
            vel_weight2 = self.velocity_branch[1](vel_feat).unsqueeze(-1).unsqueeze(-1)
        else:
            vel_weight2 = 1.0
        x = x * img_weight2 * vel_weight2

        x = self.gcn3(x)
        if self.frames and x_image is not None:
            img_feat = F.relu(self.image_branch[1][0](img_feat))
            img_weight3 = self.image_branch[2](img_feat).unsqueeze(-1).unsqueeze(-1)
        else:
            img_weight3 = 1.0
        if self.vel and x_velocity is not None:
            vel_feat = F.relu(self.velocity_branch[1][0](vel_feat))
            vel_weight3 = self.velocity_branch[2](vel_feat).unsqueeze(-1).unsqueeze(-1)
        else:
            vel_weight3 = 1.0
        x = x * img_weight3 * vel_weight3

        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        x = self.att(x).mul(x) + x     #原
        # x = self.att(x)                 # Transformer内置残差
        x = self.dropout(x)
        out = self.fc(x)
        return out
        # logits = self.fc(out)     # <-- 就是这里
        # logits = torch.clamp(logits, min=-10, max=10)  # ✅ 添加这一行
        # return logits


class STA_GCN_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, num_keypoints):
        super(STA_GCN_Unit, self).__init__()
        self.gcn_layer = GCN_Layer(in_channels, out_channels, num_keypoints)
        self.attention_layer = Attention_Layer(out_channels, num_keypoints)
        self.tcn_layer = TCN_Layer(out_channels, out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn_layer(x)
        x = self.attention_layer(x)
        x = self.tcn_layer(x)
        x = F.relu(x + res)
        return x


class GCN_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, num_keypoints):
        super(GCN_Layer, self).__init__()
        self.A = nn.Parameter(torch.eye(num_keypoints).unsqueeze(0))
        self.We = nn.Parameter(torch.ones(num_keypoints, num_keypoints))
        self.Wg = nn.Parameter(torch.randn(in_channels, out_channels))

    def forward(self, x):
        B, C, T, V = x.shape
        A = torch.sigmoid(self.We) * self.A  # [1, V, V]
        x = x.permute(0, 3, 2, 1)  # [B, V, T, C]
        x = torch.einsum('vw,bvtc->bwtc', A.squeeze(0), x)  # [B, W, T, C]
        x = torch.matmul(x, self.Wg)  # [B, W, T, out_C]
        x = x.permute(0, 3, 2, 1)  # [B, out_C, T, W]
        return x


class Attention_Layer(nn.Module):
    def __init__(self, channels, num_keypoints):
        super(Attention_Layer, self).__init__()
        self.spatial = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.temporal = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        s_att = self.spatial(x)
        t_att = self.temporal(x)
        c_att = self.channel(x)
        return x * s_att * t_att * c_att


class TCN_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TCN_Layer, self).__init__()
        self.net = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.net(x)

class TransformerVecAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, C]
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.ffn(self.norm(x))
        return x.squeeze(1)
