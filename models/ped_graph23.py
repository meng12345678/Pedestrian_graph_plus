from functools import reduce
import math
import torch
from torch import nn
import numpy as np

# 3个新的注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, channels,dropout=0.1):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [N, C, T, V]
        N, C, T, V = x.size()
        
        # 修复：在通道维度上取平均和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [N, 1, T, V]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [N, 1, T, V]
        
        # 连接特征
        combined = torch.cat([avg_out, max_out], dim=1)  # [N, 2, T, V]
        
        # 应用卷积得到注意力权重
        attention = self.sigmoid(self.conv(combined))  # [N, 1, T, V]
        attention = self.dropout(attention)
        return attention.expand_as(x)  # [N, C, T, V]


class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super(TemporalAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels // 4, 1),
            nn.InstanceNorm1d(channels // 4),
            nn.ReLU(),  
            nn.Conv1d(channels // 4, channels, 1)
        )
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [N, C, T, V]
        x_pooled = torch.mean(x, dim=3)  # [N, C, T]
        att = self.sigmoid(self.mlp(x_pooled))  # [N, C, T]
        # att = self.dropout(att)  # ⬅ Dropout 加在注意力 mask 上
        att = att.unsqueeze(-1).expand_as(x)  # [N, C, T, V]
        # print("temporal_att:", att.mean().item(), att.std().item())
        return att

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 使用更合理的降维比例
        hidden_channels = max(channels // reduction, 1)
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels)
        )
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [N, C, T, V]
        N, C, T, V = x.size()
        
        # 全局平均池化和最大池化
        avg_out = self.avg_pool(x).view(N, C)  # [N, C]
        max_out = self.max_pool(x).view(N, C)  # [N, C]
        
        # 通过共享MLP
        avg_out = self.shared_mlp(avg_out)
        max_out = self.shared_mlp(max_out)
        
        # 合并并应用sigmoid
        out = self.sigmoid(avg_out + max_out)  # [N, C]
        # out = self.dropout(out)  # ⬅ Dropout 加在权重上
        
        # 扩展到原始维度
        out = out.view(N, C, 1, 1).expand_as(x)
        # print("channel_att:", out.mean().item(), out.std().item())

        return out   

class pedMondel(nn.Module):
    # 初始化方法，接受几个参数：
    # frames: 是否使用时间帧（序列数据），用于是否处理时间序列。
    # vel: 是否使用速度特征（如果为True，则使用速度信息）。
    # seg: 是否进行分割（用于图像分割任务）。
    # h3d: 是否启用3D人体关键点数据（如果为True，则使用3D关键点数据，默认值为True）。
    # nodes: 节点数量，通常表示图中的关键点数量，默认为19。
    # n_clss: 类别数，默认为1。
    def __init__(self, frames, vel=False, seg=False, h3d=True, nodes=19, n_clss=1):
        super(pedMondel, self).__init__()
 
        self.h3d = h3d # 3D人体关键点数据是否启用。True表示启用3D数据，否则使用2D数据。
        self.frames = frames
        self.vel = vel
        self.seg = seg
        self.n_clss = n_clss
        self.ch = 4 if h3d else 3
        self.ch1, self.ch2 = 32, 64
        i_ch = 4 if seg else 3

        self.data_bn = nn.InstanceNorm1d(self.ch * nodes, affine=True, track_running_stats=False)
        bn_init(self.data_bn, 1)
        self.drop = nn.Dropout(0.25)
        A = np.stack([np.eye(nodes)] * 3, axis=0)
        
        if frames:
            self.conv0 = nn.Sequential(
                nn.Conv2d(i_ch, self.ch1, kernel_size=3, stride=1, padding=0, bias=False), 
                nn.InstanceNorm2d(self.ch1, affine=True, track_running_stats=False), nn.SiLU())
        if vel:
            self.v0 = nn.Sequential(
                nn.Conv1d(2, self.ch1, 3, bias=False),nn.InstanceNorm1d(self.ch1, affine=True, track_running_stats=False), nn.SiLU())
        # ----------------------------------------------------------------------------------------------------
        self.l1 = TCN_GCN_unit(self.ch, self.ch1, A, residual=False)

        if frames:
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.ch1, self.ch1, kernel_size=3, stride=1, padding=0, bias=False), 
                nn.InstanceNorm2d(self.ch1, affine=True, track_running_stats=False), nn.SiLU())
        if vel:
            self.v1 = nn.Sequential(
                nn.Conv1d(self.ch1, self.ch1, 3, bias=False), 
                nn.InstanceNorm1d(self.ch1, affine=True, track_running_stats=False), nn.SiLU())
        # ----------------------------------------------------------------------------------------------------
        self.l2 = TCN_GCN_unit(self.ch1, self.ch2, A)

        if frames:
            self.conv2 = nn.Sequential(
                nn.Conv2d(self.ch1, self.ch2, kernel_size=2, stride=1, padding=0, bias=False), 
                nn.InstanceNorm2d(self.ch2, affine=True, track_running_stats=False), nn.SiLU())
            
        if vel:
            self.v2 = nn.Sequential(
                nn.Conv1d(self.ch1, self.ch2, kernel_size=2, bias=False), 
                nn.InstanceNorm1d(self.ch2, affine=True, track_running_stats=False),nn.SiLU())
        # ----------------------------------------------------------------------------------------------------
        # self.l3 = TCN_GCN_unit(self.ch2, self.ch2, A)

        # if frames:
        #     self.conv3 = nn.Sequential(
        #         nn.Conv2d(self.ch2, self.ch2, kernel_size=2, stride=1, padding=0, bias=False), 
        #         nn.BatchNorm2d(self.ch2), nn.SiLU())
            
        # if vel:
        #     self.v3 = nn.Sequential(
        #         nn.Conv1d(self.ch2, self.ch2, kernel_size=2, bias=False), 
        #         nn.BatchNorm1d(self.ch2), nn.SiLU())
        # ----------------------------------------------------------------------------------------------------
        

        self.gap = nn.AdaptiveAvgPool2d(1)
        # 定义一个自适应平均池化层 (AdaptiveAvgPool2d)，
        # 它会将输入的大小池化到指定的输出大小，这里输出大小为 (1, 1)，
        # 即将每个特征图的空间维度（高和宽）压缩为1，从而获得每个特征图的全局平均值。

        self.att = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.ch2, self.ch2, bias=False),
            nn.LayerNorm(self.ch2), 
            nn.Sigmoid()
        )

        self.linear = nn.Linear(self.ch2, self.n_clss)
        # 定义一个全连接层，用于将特征图的输出映射到最终的分类数 `self.n_clss`。
        # 这个层的输入维度为 `self.ch2`（64），输出维度为 `self.n_clss`（分类数）。
        nn.init.normal_(self.linear.weight, 0, math.sqrt(2. / self.n_clss))
        # pooling sigmoid fucntion for image feature fusion
        self.pool_sigm_2d = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 对2D输入应用自适应平均池化，输出大小为 (1, 1)，
            nn.Sigmoid()
        )
        if vel:
            self.pool_sigm_1d = nn.Sequential(
                nn.AdaptiveAvgPool1d(1), # 对1D输入应用自适应平均池化，输出大小为 (1)，
                nn.Sigmoid()
            )
        
    
    def forward(self, kp, frame=None, vel=None): 

        N, C, T, V = kp.shape
        # 原始 kp ([32, 4, 32, 19]) 
        # print(f"Original kp shape: {kp.shape}")
        kp = kp.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)  # `kp` 交换维度，使得 `kp` 的形状变成 `(N, C * V, T)`，为了适应卷积操作（对关键点的时间序列进行卷积处理）
        # kp交换维度后([32, 76, 32])
        # print(f"kp after permute and view: {kp.shape}")
        kp = self.data_bn(kp) # 对 `kp` 应用批归一化（`data_bn` 是一个批归一化层），标准化输入数据
        kp = kp.view(N, C, V, T).permute(0, 1, 3, 2).contiguous() # 将 `kp` 恢复回 `(N, C, V, T)` 形状，并通过 `permute` 交换维度，以便后续处理
        # kp 恢复维度[32, 4, 32, 19]
        # print(f"kp after reshape back: {kp.shape}")

        if self.frames:
            f1 = self.conv0(frame)  # 第一次conv卷积
        if self.vel:
            v1 = self.v0(vel)  # 第一次vel卷积

        # --------------------------
        # 第一个TCN_GCN单元（内部已经按GCN→注意力→TCN的顺序处理）
        x1 = self.l1(kp)
        # x1 经过 l1层 [32, 32, 32, 19]
        # print(f"x1 after l1: {x1.shape}")

        if self.frames:
            f1 = self.conv1(f1)   
            x1 = x1 * self.pool_sigm_2d(f1)# 对 `f1` 应用池化并使用 Sigmoid 激活函数融合特征，再与 `x1` 相乘，作为加权融合的操作
            # x1 经过 frame 融合 ([32, 32, 32, 19])
            # print(f"x1 after frame fusion: {x1.shape}")
        if self.vel:   
            v1 = self.v1(v1)
            x1 = x1 * self.pool_sigm_1d(v1).unsqueeze(-1) # 对 `v1` 应用池化并使用 Sigmoid 激活函数融合特征，再与 `x1` 相乘，作为加权融合的操作
            # x1 经过 velocity 融合: ([32, 32, 32, 19])
            # print(f"x1 after velocity fusion: {x1.shape}")      
        # --------------------------      
         
        # --------------------------
        # 第二个TCN_GCN单元（内部已经按GCN→注意力→TCN的顺序处理）
        x1 = self.l2(x1)
        # x1 经过 l2层: torch.Size([32, 64, 32, 19])
        # print(f"x1 after l2: {x1.shape}")

        if self.frames:
            f1 = self.conv2(f1) 
            x1 = x1 * self.pool_sigm_2d(f1)
            # x1 经过 第二次 frame 融合: torch.Size([32, 64, 32, 19])
            # print(f"x1 after second frame fusion: {x1.shape}")
        if self.vel:  
            v1 = self.v2(v1)
            x1 = x1 * self.pool_sigm_1d(v1).unsqueeze(-1)  # 第二次 velocity 融合
            # x1 经过 第二次 velocity 融合: torch.Size([32, 64, 32, 19])
            # print(f"x1 after second velocity fusion: {x1.shape}")              
        # --------------------------
  
        # x1 = self.l3(x1)
        # if self.frames:
        #     f1 = self.conv3(f1) 
        #     x1 = x1.mul(self.pool_sigm_2d(f1))
        # if self.vel:  
        #     v1 = self.v3(v1)
        #     x1 = x1.mul(self.pool_sigm_1d(v1).unsqueeze(-1))
        # --------------------------

        x1 = self.gap(x1).squeeze(-1)
        x1 = x1.squeeze(-1)
        x1 = self.att(x1).mul(x1) + x1
        x1 = self.drop(x1)
        x1 = self.linear(x1)

        return x1


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive # 根据是否自适应决定邻接矩阵的处理方式
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)  # 创建一个可训练的参数 PA
        else:
            self.A = torch.autograd.Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False) # 使用固定的邻接矩阵 A
        
        # 定义一个模块列表 conv_d，用于存储每个子集对应的卷积层
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1)) # 每个子集对应一个 1x1 的卷积

         # 如果输入和输出通道数不一致，则需要一个卷积和批归一化来调整维度
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.InstanceNorm1d, nn.InstanceNorm2d)):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)
    
    def L2_norm(self, A):
        # A: (N, V, V) 其中 N 是批量大小，V 是图的节点数
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            A = self.L2_norm(A)
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):

            A1 = A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y

class TCN_GCN_unit(nn.Module): 
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, 
                    #  gcn_dropout=0.1, 
                    #  attention_dropout=0.1, 
                    #  tcn_dropout=0.1, 
                    #  final_dropout=0.3
                     ):
        super(TCN_GCN_unit, self).__init__()
        # 定义第一个图卷积层
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)  
        # 添加三个注意力模块
        self.spatial_attention = SpatialAttention(out_channels)
        self.temporal_attention = TemporalAttention(out_channels)
        #  dropout=0.1)
        
        self.channel_attention = ChannelAttention(out_channels)
        # dropout=0.1)
        
        # 定义第一个时间卷积层
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)  
        self.relu = nn.ReLU(inplace=True)

        # # 为不同位置添加不同的dropout层
        # self.gcn_dropout = nn.Dropout(gcn_dropout)           # GCN后的dropout
        # self.attention_dropout = nn.Dropout(attention_dropout) # 通道注意力后的dropout
        # self.tcn_dropout = nn.Dropout(tcn_dropout)           # TCN后的dropout
        # self.final_dropout = nn.Dropout(final_dropout)       # 最后sum+relu后的dropout

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        # 1. 先应用GCN
        gcn_out = self.gcn1(x)  
        # # 在GCN后添加dropout（较低的dropout率，保持更多信息）
        # gcn_out = self.gcn_dropout(gcn_out)
        
        # 2. 然后依次应用三个注意力模块
        # 空间注意力
        spatial_att = self.spatial_attention(gcn_out)
        gcn_out = gcn_out * spatial_att
        
        # 时间注意力
        temporal_att = self.temporal_attention(gcn_out)
        gcn_out = gcn_out * temporal_att

        # 通道注意力
        channel_att = self.channel_attention(gcn_out)
        gcn_out = gcn_out * channel_att
        # # 在通道注意力后添加dropout（中等dropout率）
        # gcn_out = self.attention_dropout(gcn_out)

        # 3. 最后应用TCN
        y = self.tcn1(gcn_out)
        # # 在TCN后添加dropout（较高的dropout率，防止过拟合）
        # y = self.tcn_dropout(y)
        
        # 添加残差连接并应用ReLU
        y = self.relu(y + self.residual(x))
        # # 在最后的sum+relu后添加dropout（较低的dropout率，保持输出稳定）
        # y = self.final_dropout(y)
        return y
