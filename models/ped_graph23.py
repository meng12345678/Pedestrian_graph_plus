from functools import reduce
import math
import torch
from torch import nn
import numpy as np

# 3个新的注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        return self.sigmoid(avg_out)

class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super(TemporalAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 这里假设输入是 4D 张量 (batch_size, channels, time_steps, nodes)
        # 我们先 reshape 成 3D 张量 (batch_size, channels * nodes, time_steps)
        N, C, T, V = x.size()

         # 重新调整维度为 (batch_size, channels * nodes, time_steps)
        x = x.view(N, C * V, T)  # 将输入从 4D 转换为 3D 张量
        avg_out = self.avg_pool(x)  # 对时间维度进行平均池化
        avg_out = avg_out.unsqueeze(-1)  # 添加额外的维度，以便后续广播
     
        avg_out = avg_out.view(avg_out.shape[0], avg_out.shape[1], 1)  # 变成 [32, 608, 1]

        # 扩展到 (batch_size, channels * nodes, time_steps)
        avg_out = avg_out.expand_as(x)  # 将池化输出扩展为与输入张量相同的时间维度
        # 恢复成四维：[batch_size, channels, time_steps, nodes]
        avg_out = avg_out.reshape(N, C, T, V)
        return self.sigmoid(avg_out)

class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(channels, channels)  
        self.fc2 = nn.Linear(channels, channels)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=(2, 3), keepdim=False)  # 对 H 和 W 维度求平均
        # print("avg_out shape:", avg_out.shape)  # 打印avg_out的形状
        avg_out = avg_out.view(avg_out.size(0), -1)  # 展平为 (batch_size, channels)
        # print("avg_out after reshape shape:", avg_out.shape)  # 打印 reshape 后的 avg_out 形状
        fc_out = self.fc2(self.fc1(avg_out))  # fc1 -> fc2

        # try:
        #     # 确保fc1的输入输出维度一致
        #     fc1_out = self.fc1(avg_out)
        #     print("fc1 output shape:", fc1_out.shape)
        # except Exception as e:
        #     print(f"Error in fc1: {e}")
        #     fc1_out = torch.zeros_like(avg_out)  # 设置默认值，避免后续报错

        # try:
        #     # fc2的输入和输出维度应与fc1一致
        #     fc2_out = self.fc2(fc1_out)
        #     print("fc2 output shape:", fc2_out.shape)
        # except Exception as e:
        #     print(f"Error in fc2: {e}")
        #     fc2_out = torch.zeros_like(fc1_out)  # 设置默认值，避免后续报错

        # return self.sigmoid(fc2_out).view(x.size(0), x.size(1), 1, 1)  # 返回输出
        
        return self.sigmoid(fc_out).view(-1, avg_out.size(1), 1, 1)  # 输出形状应该是 (batch_size, channels, 1, 1)

class pedMondel(nn.Module):
    # 初始化方法，接受几个参数：
    # frames: 是否使用时间帧（序列数据），用于是否处理时间序列。
    # vel: 是否使用速度特征（如果为True，则使用速度信息）。
    # seg: 是否进行分割（用于图像分割任务）。
    # h3d: 是否启用3D人体关键点数据（如果为True，则使用3D关键点数据，默认值为True）。
    # nodes: 节点数量，通常表示图中的关键点数量，默认为19。
    # n_clss: 类别数，默认为1。
    def __init__(self, frames, vel=False, seg=False, h3d=False, nodes=19, n_clss=1):
        super(pedMondel, self).__init__()
 
        self.h3d = h3d # 3D人体关键点数据是否启用。True表示启用3D数据，否则使用2D数据。
        self.frames = frames
        self.vel = vel
        self.seg = seg
        self.n_clss = n_clss
        self.ch = 4 if h3d else 3
        self.ch1, self.ch2 = 32, 64
        i_ch = 4 if seg else 3

        self.data_bn = nn.BatchNorm1d(self.ch * nodes)  # 创建一个批归一化层，用于1D数据，输入通道为ch*nodes
        bn_init(self.data_bn, 1) # 初始化批归一化层，通常会设置初始化方法
        self.drop = nn.Dropout(0.25)  
        A = np.stack([np.eye(nodes)] * 3, axis=0) # 创建一个3x(nodes x nodes)的单位矩阵（邻接矩阵），用于图神经网络或其他图结构计算。
        
        if frames: # 如果使用时间帧（frames为True）
            self.conv0 = nn.Sequential(
                nn.Conv2d(i_ch, self.ch1, kernel_size=3, stride=1, padding=0, bias=False), # 定义第一个2D卷积层，输入通道为i_ch（3或4），输出通道为self.ch1（32），卷积核大小为3x3，不使用偏置
                nn.BatchNorm2d(self.ch1), nn.SiLU())
        if vel: # 如果使用速度特征（vel为True）  
            self.v0 = nn.Sequential(
                nn.Conv1d(2, self.ch1, 3, bias=False), 
                nn.BatchNorm1d(self.ch1), nn.SiLU()) # 定义一个1D卷积层，输入通道为2（速度信息x和y），输出通道为self.ch1（32），卷积核大小为3。
        # ----------------------------------------------------------------------------------------------------
        # 创建一个TCN_GCN_unit（已经包含了按GCN→注意力→TCN顺序的处理），
        # 输入通道为self.ch（由h3d决定，3D数据为4，2D为3），输出通道为self.ch1（32），
        # 使用的邻接矩阵为A（图的邻接矩阵），residual=False表示不使用残差连接。
        self.l1 = TCN_GCN_unit(self.ch, self.ch1, A, residual=False)

        if frames:
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.ch1, self.ch1, kernel_size=3, stride=1, padding=0, bias=False), # 定义第二个2D卷积层，输入和输出通道都是self.ch1（32），卷积核大小为3x3。
                nn.BatchNorm2d(self.ch1), nn.SiLU())
        if vel:
            self.v1 = nn.Sequential(
                nn.Conv1d(self.ch1, self.ch1, 3, bias=False),  # 定义第二个1D卷积层，输入和输出通道为self.ch1（32），卷积核大小为3。
                nn.BatchNorm1d(self.ch1), nn.SiLU())
        # ----------------------------------------------------------------------------------------------------
        # 创建一个TCN_GCN_unit（已经包含了按GCN→注意力→TCN顺序的处理），
        # 输入通道为self.ch1（32），输出通道为self.ch2（64），
        # 使用的邻接矩阵为A（图的邻接矩阵）。
        self.l2 = TCN_GCN_unit(self.ch1, self.ch2, A)

        if frames:
            self.conv2 = nn.Sequential(
                nn.Conv2d(self.ch1, self.ch2, kernel_size=2, stride=1, padding=0, bias=False),  # 定义第三个2D卷积层，输入通道为self.ch1（32），输出通道为self.ch2（64），卷积核大小为2x2。
                nn.BatchNorm2d(self.ch2), nn.SiLU())
            
        if vel:
            self.v2 = nn.Sequential(
                nn.Conv1d(self.ch1, self.ch2, kernel_size=2, bias=False), # 定义第三个1D卷积层，输入通道为self.ch1（32），输出通道为self.ch2（64），卷积核大小为2。
                nn.BatchNorm1d(self.ch2), nn.SiLU())
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
        # x1 经过池化 gap: torch.Size([32, 64, 1])
        # print(f"x1 after gap: {x1.shape}")
        x1 = x1.squeeze(-1)
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
                              stride=(stride, 1)) # 定义二维卷积层，卷积核大小为 (kernel_size, 1)，仅在高度方向使用 padding 和 stride

        self.bn = nn.BatchNorm2d(out_channels)
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
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
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
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True):
        super(TCN_GCN_unit, self).__init__()
        # 定义第一个图卷积层
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)  
        # 添加三个注意力模块
        self.spatial_attention = SpatialAttention(out_channels)
        self.temporal_attention = TemporalAttention(out_channels)
        self.channel_attention = ChannelAttention(out_channels)
        # 定义第一个时间卷积层
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)  
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        # 1. 先应用GCN
        gcn_out = self.gcn1(x)
        
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
        
        # 3. 最后应用TCN
        y = self.tcn1(gcn_out)
        
        # 添加残差连接并应用ReLU
        y = self.relu(y + self.residual(x))
        return y
