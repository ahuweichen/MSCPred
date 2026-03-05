import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
from torch.nn.utils.weight_norm import weight_norm  # 导入权重归一化工具

# BANLayer类定义了一个双线性注意力网络层
class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, dropout, k, act='ReLU'):
        super(BANLayer, self).__init__()  # 调用父类初始化方法
        self.c = 32  # 定义常量c
        self.k = k  # 头数k
        self.v_dim = v_dim  # 视觉特征维度
        self.q_dim = q_dim  # 问题特征维度
        self.h_dim = h_dim  # 隐藏层维度
        self.h_out = h_out  # 输出维度
        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)  # 视觉特征转换网络
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)  # 问题特征转换网络

        # 如果k大于1，则使用平均池化
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        # 根据h_out的大小决定是否使用参数矩阵和偏置
        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())  # 初始化参数矩阵
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())  # 初始化偏置
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)  # 使用线性层

        self.bn = nn.BatchNorm1d(h_dim)  # 批归一化层

    # 注意力池化函数
    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))  # 计算融合后的logits
        if 1 < self.k:  # 如果k大于1，进行额外处理
            fusion_logits = fusion_logits.unsqueeze(1)  # 增加一个维度
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # 平均池化后乘以k
        return fusion_logits

    # 前向传播函数
    def forward(self, v, q, softmax=False):
        if self.h_out <= self.c:  # 当输出维度小于等于c时
            v_ = self.v_net(v)  # 通过视觉网络
            q_ = self.q_net(q)  # 通过问题网络
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias  # 计算注意力图
        else:  # 当输出维度大于c时
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)  # 通过视觉网络并调整维度
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)  # 通过问题网络并调整维度
            d_ = torch.matmul(v_, q_)  # 矩阵相乘
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # 通过线性层
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # 调整维度顺序
        if softmax:  # 如果需要softmax（未实现）
            pass
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])  # 对第一个注意力图进行池化
        for i in range(1, self.h_out):  # 对其他注意力图进行池化，并累加结果
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)  # 应用批归一化
        return logits, att_maps  # 返回logits和注意力图

# FCNet类定义了一个非线性的全连接网络
class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()
        layers = []  # 存储网络层
        for i in range(len(dims) - 2):  # 循环构建除最后一层外的所有层
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:  # 如果有dropout
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))  # 添加带权重归一化的线性层
            if '' != act:  # 如果指定了激活函数
                layers.append(getattr(nn, act)())  # 添加激活函数
        if 0 < dropout:  # 如果有dropout
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))  # 添加最后一层线性层
        if '' != act:  # 如果指定了激活函数
            layers.append(getattr(nn, act)())  # 添加激活函数
        self.main = nn.Sequential(*layers)  # 将所有层组合成一个序列模型

    def forward(self, x):
        return self.main(x)  # 前向传播

# BCNet类定义了一个非线性的双线性连接网络
class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/bc.py
    """
    def __init__(self, v_dim, q_dim, h_dim, h_out=None, act='ReLU', dropout=[.5, .7], k=3):
        super(BCNet, self).__init__()
        self.c = 32  # 常量c
        self.k = k  # 头数k
        self.v_dim = v_dim  # 视觉特征维度
        self.q_dim = q_dim  # 问题特征维度
        self.h_dim = h_dim  # 隐藏层维度
        self.h_out = h_out  # 输出维度
        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])  # 视觉特征转换网络
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])  # 问题特征转换网络
        self.dropout = nn.Dropout(dropout[1])  # 注意力机制中的Dropout
        if 1 < k:  # 如果头数k大于1
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)  # 定义平均池化层
        if None == h_out:  # 如果没有指定输出维度
            pass
        elif h_out <= self.c:  # 如果输出维度小于等于c
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())  # 初始化参数矩阵
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())  # 初始化偏置
        else:  # 如果输出维度大于c
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)  # 定义线性层

    def forward(self, v, q):
        if self.h_out <= self.c:  # 如果输出维度小于等于c
            v_ = self.dropout(self.v_net(v))  # 通过视觉网络并应用Dropout
            q_ = self.q_net(q)  # 通过问题网络
            logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias  # 计算logits
            return logits  # 返回logits
        else:  # 如果输出维度大于c
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)  # 通过视觉网络并调整维度
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)  # 通过问题网络并调整维度
            d_ = torch.matmul(v_, q_)  # 矩阵相乘
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # 通过线性层
        return logits.transpose(2, 3).transpose(1, 2)  # 调整logits维度并返回

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v)  # 通过视觉网络
        q_ = self.q_net(q)  # 通过问题网络
        logits = torch.einsum('bvk,bvq,bqk->bk', (v_, w, q_))  # 计算logits
        if 1 < self.k:  # 如果头数k大于1
            logits = logits.unsqueeze(1)  # 增加一个维度
            logits = self.p_net(logits).squeeze(1) * self.k  # 平均池化后乘以k
        return logits  # 返回logits

# BiAttention类定义了一个双向注意力机制
class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.5, .7]):
        super(BiAttention, self).__init__()
        self.glimpse = glimpse  # 观察次数
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3),
                                  name='h_mat', dim=None)  # 定义BCNet作为logits计算

    def forward(self, v, q, v_mask=True):
        p, logits = self.forward_all(v, q, v_mask)  # 调用forward_all计算概率和logits
        return p, logits  # 返回概率和logits

    def forward_all(self, v, q, v_mask=True, logit=False, mask_with=-float('inf')):
        v_num = v.size(1)  # 视觉特征数量
        q_num = q.size(1)  # 问题特征数量
        logits = self.logits(v, q)  # 计算logits
        if v_mask:  # 如果需要对视觉特征进行掩码
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())  # 创建掩码
            logits.data.masked_fill_(mask.data, mask_with)  # 应用掩码
        if not logit:  # 如果不需要返回原始logits
            p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)  # 计算softmax
            return p.view(-1, self.glimpse, v_num, q_num), logits  # 返回概率和logits
        return logits  # 只返回logits

if __name__ == '__main__':
    net = BANLayer(1024, 1024, 1024, 2).cuda()  # 创建BANLayer实例并移到GPU
    x = torch.Tensor(512, 36, 1024).cuda()  # 创建输入张量x并移到GPU
    y = torch.Tensor(512, 14, 1024).cuda()  # 创建输入张量y并移到GPU
    out, _ = net.forward(x, y)  # 前向传播
    print(out.shape)  # 打印输出形状