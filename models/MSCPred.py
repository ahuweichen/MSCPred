#!/usr/bin/env python
# coding=UTF-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.ban import BANLayer  # BANLayer是外部定义的Bilinear Attention Networks层


class TextCNN_block1(nn.Module):
    def __init__(self, vocab_size, embedding_dim_DLM, embedding_dim_seq, DLM_seq_len, sequence_length, n_filters,
                 filter_sizes, output_dim, dropout):
        super(TextCNN_block1, self).__init__()
        # 初始化多分枝卷积层
        self.convs1 = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim_DLM,  # 输入通道数
                                               out_channels=n_filters,  # 输出通道数
                                               kernel_size=fs,  # 卷积核大小
                                               padding='same')  # 使用'same'填充以保持输出长度与输入相同
                                     for fs in filter_sizes])  # 对每个filter size创建一个卷积层
        # 定义全连接层
        self.fc1 = nn.Linear(1920, 640)  # 将卷积后的特征映射到512维
        self.fc = nn.Sequential(
            nn.Linear(640, 32),  # 线性层，将512维特征映射到32维
            nn.Mish(),  # Mish激活函数
            nn.Dropout(),  # Dropout层，防止过拟合
            nn.Linear(32, 8),  # 线性层，将32维特征映射到8维
            nn.Mish(),  # Mish激活函数
            nn.Linear(8, output_dim)  # 最终输出层，将8维特征映射到output_dim
        )
        # 定义Dropout层和激活函数
        self.dropout1 = nn.Dropout(dropout)
        self.Mish1 = nn.Mish()
        # 定义批归一化层
        self.batchnorm1 = nn.BatchNorm1d(640)

    def forward(self, DLM_fea, seq_data, chr_data):
        # 对DLM特征进行维度变换，从[batch_size, sequence_length, embedding_dim]变为[batch_size, embedding_dim, sequence_length]
        DLM_embedded = DLM_fea.permute(0, 2, 1)
        # 应用卷积层并使用Mish激活函数
        DLM_conved = [self.Mish1(conv(DLM_embedded)) for conv in self.convs1]
        # 池化层，使用最大池化
        DLM_pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in DLM_conved]
        # 多分支线性展开
        DLM_flatten = [pool.contiguous().view(pool.size(0), -1) for pool in DLM_pooled]
        # 将各分支连接在一起
        DLM_cat = self.dropout1(torch.cat(DLM_flatten, dim=1))
        # 使用线性层进行维度变换，并应用批归一化
        DLM_cat_i = self.fc1(DLM_cat)
        DLM_cat_i = self.batchnorm1(DLM_cat_i)

        # 输出特征并分类
        return self.fc(DLM_cat_i), DLM_cat_i  # 返回最终分类结果和中间特征


class TextCNN_block2(nn.Module):
    def __init__(self, vocab_size, embedding_dim_DLM, embedding_dim_seq, DLM_seq_len, sequence_length, n_filters,
                 filter_sizes, output_dim, dropout):
        super(TextCNN_block2, self).__init__()
        # 初始化多分枝卷积层
        self.convs2 = nn.ModuleList([nn.Conv1d(in_channels=4,  # 输入通道数
                                               out_channels=n_filters,  # 输出通道数
                                               kernel_size=fs,  # 卷积核大小
                                               padding='same')  # 使用'same'填充以保持输出长度与输入相同
                                     for fs in filter_sizes])
        # 定义全连接层
        self.fc2 = nn.Linear(1920, 512)  # 将卷积后的特征映射到512维
        self.fc6 = nn.Linear(919, 512)  # 将染色质数据映射到512维
        self.fc3 = nn.Sequential(
            nn.Linear(512, 32),  # 线性层，将512维特征映射到32维
            nn.Mish(),  # Mish激活函数
            nn.Dropout(),  # Dropout层，防止过拟合
            nn.Linear(32, 8),  # 线性层，将32维特征映射到8维
            nn.Mish(),  # Mish激活函数
            nn.Linear(8, output_dim)  # 最终输出层，将8维特征映射到output_dim
        )
        # 定义Dropout层和激活函数
        self.dropout2 = nn.Dropout(dropout)
        self.Mish2 = nn.Mish()
        # 定义批归一化层
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm4 = nn.BatchNorm1d(919)
        # 定义Bilinear Attention Networks层
        self.ban1 = BANLayer(512, 512, 512, 2, 0.1, 6)

    def forward(self, DLM_fea, seq_data, chr_data):
        # 对序列数据进行维度变换
        seq_embedded2 = seq_data.permute(0, 2, 1)
        # 应用卷积层并使用Mish激活函数
        seq_conved = [self.Mish2(conv(seq_embedded2)) for conv in self.convs2]
        # 池化层，使用最大池化
        seq_pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in seq_conved]
        # 多分支线性展开
        seq_flatten = [pool.contiguous().view(pool.size(0), -1) for pool in seq_pooled]
        # 将各分支连接在一起
        seq_cat = self.dropout2(torch.cat(seq_flatten, dim=1))
        # 使用线性层进行维度变换，并应用批归一化
        seq_cat_i = self.fc2(seq_cat)
        seq_cat_i = self.batchnorm2(seq_cat_i)

        # 对染色质相关数据进行批归一化和线性变换
        fea_data = self.batchnorm4(chr_data)
        fea = self.fc6(fea_data)

        # 将序列数据和染色质相关数据融合在一起
        fusion, att_weight = self.ban1(seq_cat_i.unsqueeze(1), fea.unsqueeze(1))

        # 输出特征并分类
        return self.fc3(fusion), fusion  # 返回最终分类结果和中间特征


class TextCNN_block3(nn.Module):
    def __init__(self, vocab_size, embedding_dim_DLM, embedding_dim_seq, DLM_seq_len, sequence_length, n_filters,
                 filter_sizes, output_dim, dropout):
        super(TextCNN_block3, self).__init__()
        # 初始化多分枝卷积层
        self.convs1 = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim_DLM,  # 输入通道数
                                               out_channels=n_filters,  # 输出通道数
                                               kernel_size=fs,  # 卷积核大小
                                               padding='same')  # 使用'same'填充以保持输出长度与输入相同
                                     for fs in filter_sizes])  # 对每个filter size创建一个卷积层
        # 定义全连接层
        self.fc1 = nn.Linear(1920, 128)  # 将卷积后的特征映射到512维
        self.fc = nn.Sequential(
            nn.Linear(128, 32),  # 线性层，将512维特征映射到32维
            nn.Mish(),  # Mish激活函数
            nn.Dropout(),  # Dropout层，防止过拟合
            nn.Linear(32, 8),  # 线性层，将32维特征映射到8维
            nn.Mish(),  # Mish激活函数
            nn.Linear(8, output_dim)  # 最终输出层，将8维特征映射到output_dim
        )
        # 定义Dropout层和激活函数
        self.dropout1 = nn.Dropout(dropout)
        self.Mish1 = nn.Mish()
        # 定义批归一化层
        self.batchnorm1 = nn.BatchNorm1d(128)

    def forward(self, DLM_fea, seq_data, chr_data):
        # 对DLM特征进行维度变换，从[batch_size, sequence_length, embedding_dim]变为[batch_size, embedding_dim, sequence_length]
        DLM_embedded = DLM_fea.permute(0, 2, 1)
        # 应用卷积层并使用Mish激活函数
        DLM_conved = [self.Mish1(conv(DLM_embedded)) for conv in self.convs1]
        # 池化层，使用最大池化
        DLM_pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in DLM_conved]
        # 多分支线性展开
        DLM_flatten = [pool.contiguous().view(pool.size(0), -1) for pool in DLM_pooled]
        # 将各分支连接在一起
        DLM_cat = self.dropout1(torch.cat(DLM_flatten, dim=1))
        # 使用线性层进行维度变换，并应用批归一化
        DLM_cat_i = self.fc1(DLM_cat)
        DLM_cat_i = self.batchnorm1(DLM_cat_i)

        # 输出特征并分类
        return self.fc(DLM_cat_i), DLM_cat_i  # 返回最终分类结果和中间特征


class TextCNN_block4(nn.Module):
    def __init__(self, vocab_size, embedding_dim_DLM, embedding_dim_seq, DLM_seq_len, sequence_length, n_filters,
                 filter_sizes, output_dim, dropout):
        super(TextCNN_block4, self).__init__()
        # 初始化多分枝卷积层
        self.convs2 = nn.ModuleList([nn.Conv1d(in_channels=12,  # 输入通道数
                                               out_channels=n_filters,  # 输出通道数
                                               kernel_size=fs,  # 卷积核大小
                                               padding='same')  # 使用'same'填充以保持输出长度与输入相同
                                     for fs in filter_sizes])  # 对每个filter size创建一个卷积层
        # 定义全连接层
        self.fc2 = nn.Linear(1920, 512)  # 将卷积后的特征映射到512维
        self.fc6 = nn.Linear(919, 512)  # 将染色质数据映射到512维
        self.fc3 = nn.Sequential(
            nn.Linear(128, 32),  # 线性层，将512维特征映射到32维
            nn.Mish(),  # Mish激活函数
            nn.Dropout(),  # Dropout层，防止过拟合
            nn.Linear(32, 8),  # 线性层，将32维特征映射到8维
            nn.Mish(),  # Mish激活函数
            nn.Linear(8, output_dim)  # 最终输出层，将8维特征映射到output_dim
        )
        # 定义Dropout层和激活函数
        self.dropout2 = nn.Dropout(dropout)
        self.Mish2 = nn.Mish()
        # 定义批归一化层
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm4 = nn.BatchNorm1d(919)
        # 定义Bilinear Attention Networks层
        self.ban1 = BANLayer(512, 512, 128, 2, 0.1, 6)

    def forward(self, DLM_fea, seq_data, chr_data):
        # 对序列数据进行维度变换
        # 对DLM特征进行维度变换，从[batch_size, sequence_length, embedding_dim]变为[batch_size, embedding_dim, sequence_length]
        DLM_embedded = DLM_fea.permute(0, 1, 2)
        # 应用卷积层并使用Mish激活函数
        DLM_conved = [self.Mish2(conv(DLM_embedded)) for conv in self.convs2]
        # 池化层，使用最大池化
        DLM_pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in DLM_conved]
        # 多分支线性展开
        DLM_flatten = [pool.contiguous().view(pool.size(0), -1) for pool in DLM_pooled]
        # 将各分支连接在一起
        DLM_cat = self.dropout2(torch.cat(DLM_flatten, dim=1))
        # 使用线性层进行维度变换，并应用批归一化
        DLM_cat_i = self.fc2(DLM_cat)
        DLM_cat_i = self.batchnorm2(DLM_cat_i)

        # 对染色质相关数据进行批归一化和线性变换
        fea_data = self.batchnorm4(chr_data)
        fea = self.fc6(fea_data)

        # 将序列数据和染色质相关数据融合在一起
        fusion, att_weight = self.ban1(DLM_cat_i.unsqueeze(1), fea.unsqueeze(1))

        # 输出特征并分类
        return self.fc3(fusion), fusion  # 返回最终分类结果和中间特征


class TextCNN_block10(nn.Module):
    def __init__(self, vocab_size, embedding_dim_DLM, embedding_dim_seq, DLM_seq_len, sequence_length, n_filters,
                 filter_sizes, output_dim, dropout):
        super(TextCNN_block10, self).__init__()
        # 初始化多分枝卷积层
        self.convs1 = nn.ModuleList([nn.Conv1d(in_channels=12,  # 输入通道数
                                               out_channels=n_filters,  # 输出通道数
                                               kernel_size=fs,  # 卷积核大小
                                               padding='same')  # 使用'same'填充以保持输出长度与输入相同
                                     for fs in filter_sizes])  # 对每个filter size创建一个卷积层
        # 定义全连接层
        self.fc1 = nn.Linear(1920, 128)  # 将卷积后的特征映射到512维
        self.fc = nn.Sequential(
            nn.Linear(128, 32),  # 线性层，将512维特征映射到32维
            nn.Mish(),  # Mish激活函数
            nn.Dropout(),  # Dropout层，防止过拟合
            nn.Linear(32, 8),  # 线性层，将32维特征映射到8维
            nn.Mish(),  # Mish激活函数
            nn.Linear(8, output_dim)  # 最终输出层，将8维特征映射到output_dim
        )
        # 定义Dropout层和激活函数
        self.dropout1 = nn.Dropout(dropout)
        self.Mish1 = nn.Mish()
        # 定义批归一化层
        self.batchnorm1 = nn.BatchNorm1d(128)

    def forward(self, DLM_fea, seq_data, chr_data):
        # 对DLM特征进行维度变换，从[batch_size, sequence_length, embedding_dim]变为[batch_size, embedding_dim, sequence_length]
        DLM_embedded = DLM_fea.permute(0, 1, 2)
        # 应用卷积层并使用Mish激活函数
        DLM_conved = [self.Mish1(conv(DLM_embedded)) for conv in self.convs1]
        # 池化层，使用最大池化
        DLM_pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in DLM_conved]
        # 多分支线性展开
        DLM_flatten = [pool.contiguous().view(pool.size(0), -1) for pool in DLM_pooled]
        # 将各分支连接在一起
        DLM_cat = self.dropout1(torch.cat(DLM_flatten, dim=1))
        # 使用线性层进行维度变换，并应用批归一化
        DLM_cat_i = self.fc1(DLM_cat)
        DLM_cat_i = self.batchnorm1(DLM_cat_i)

        # 输出特征并分类
        return self.fc(DLM_cat_i), DLM_cat_i  # 返回最终分类结果和中间特征


class TextCNN_block5(nn.Module):
    def __init__(self, vocab_size, embedding_dim_DLM, embedding_dim_seq, DLM_seq_len, sequence_length, n_filters,
                 filter_sizes, output_dim, dropout):
        super(TextCNN_block5, self).__init__()
        # 初始化多分枝卷积层
        self.convs1 = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim_DLM,  # 输入通道数
                                               out_channels=n_filters,  # 输出通道数
                                               kernel_size=fs,  # 卷积核大小
                                               padding='same')  # 使用'same'填充以保持输出长度与输入相同
                                     for fs in filter_sizes])  # 对每个filter size创建一个卷积层
        # 定义全连接层
        self.fc1 = nn.Linear(2112, 128)  # 将卷积后的特征映射到512维
        self.fc = nn.Sequential(
            nn.Linear(128, 32),  # 线性层，将512维特征映射到32维
            nn.Mish(),  # Mish激活函数
            nn.Dropout(),  # Dropout层，防止过拟合
            nn.Linear(32, 8),  # 线性层，将32维特征映射到8维
            nn.Mish(),  # Mish激活函数
            nn.Linear(8, output_dim)  # 最终输出层，将8维特征映射到output_dim
        )
        # 定义Dropout层和激活函数
        self.dropout1 = nn.Dropout(dropout)
        self.Mish1 = nn.Mish()
        # 定义批归一化层
        self.batchnorm1 = nn.BatchNorm1d(128)

    def forward(self, DLM_fea, seq_data, chr_data):
        # 对DLM特征进行维度变换，从[batch_size, sequence_length, embedding_dim]变为[batch_size, embedding_dim, sequence_length]
        DLM_embedded = DLM_fea.permute(0, 2, 1)
        # 应用卷积层并使用Mish激活函数
        DLM_conved = [self.Mish1(conv(DLM_embedded)) for conv in self.convs1]
        # 池化层，使用最大池化
        DLM_pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in DLM_conved]
        # 多分支线性展开
        DLM_flatten = [pool.contiguous().view(pool.size(0), -1) for pool in DLM_pooled]
        # 将各分支连接在一起
        DLM_cat = self.dropout1(torch.cat(DLM_flatten, dim=1))
        # 使用线性层进行维度变换，并应用批归一化
        DLM_cat_i = self.fc1(DLM_cat)
        DLM_cat_i = self.batchnorm1(DLM_cat_i)

        # 输出特征并分类
        return self.fc(DLM_cat_i), DLM_cat_i  # 返回最终分类结果和中间特征


class TextCNN_block6(nn.Module):
    def __init__(self, vocab_size, embedding_dim_DLM, embedding_dim_seq, DLM_seq_len, sequence_length, n_filters,
                 filter_sizes, output_dim, dropout):
        super(TextCNN_block6, self).__init__()
        # 初始化线性层，将输入通道数从768降到128
        self.linear_DLM = nn.Linear(768, 128)

        # 初始化多分枝卷积层（用于DLM特征）
        self.convs2 = nn.ModuleList([
            nn.Conv1d(
                in_channels=128,  # 修改为128以匹配降维后的输入通道数
                out_channels=n_filters,
                kernel_size=fs,
                padding='same'
            )
            for fs in filter_sizes
        ])

        # 初始化多分枝卷积层（用于染色质特征）
        self.convs_chr = nn.ModuleList([
            nn.Conv1d(
                in_channels=128,  # 输入通道数（假设染色质数据为一维）
                out_channels=n_filters,
                kernel_size=fs,
                padding='same'
            )
            for fs in filter_sizes
        ])

        # 定义全连接层（分别为DLM和chr定义）
        self.fc2_DLM = nn.Linear(2112, 512)
        self.fc2_chr = nn.Linear(1920, 512)

        self.fc3 = nn.Sequential(
            nn.Linear(256, 32),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(32, 8),
            nn.Mish(),
            nn.Linear(8, output_dim)
        )

        # 定义Dropout层和激活函数
        self.dropout2 = nn.Dropout(dropout)
        self.Mish2 = nn.Mish()

        # 定义批归一化层
        self.batchnorm2_DLM = nn.BatchNorm1d(512)
        self.batchnorm2_chr = nn.BatchNorm1d(512)
        self.batchnorm_chr = nn.BatchNorm1d(1920)

        # 定义Bilinear Attention Networks层
        self.ban1 = BANLayer(512, 512, 256, 2, 0.1, 6)

    def forward(self, DLM_fea, seq_data, DLM_fea1):
        # 对DLM特征进行维度变换，从[batch_size, sequence_length, embedding_dim]变为[batch_size, embedding_dim, sequence_length]
        DLM_embedded = DLM_fea.permute(0, 2, 1)

        # 使用线性层将通道数从768降到128
        DLM_embedded = self.linear_DLM(DLM_embedded.permute(0, 2, 1)).permute(0, 2, 1)

        # 应用卷积层并使用Mish激活函数
        DLM_conved = [self.Mish2(conv(DLM_embedded)) for conv in self.convs2]

        # 池化层，使用最大池化
        DLM_pooled = [
            F.max_pool1d(conv, math.ceil(conv.shape[2] // 10))
            for conv in DLM_conved
        ]

        # 多分支线性展开
        DLM_flatten = [
            pool.contiguous().view(pool.size(0), -1)
            for pool in DLM_pooled
        ]

        # 将各分支连接在一起
        DLM_cat = self.dropout2(torch.cat(DLM_flatten, dim=1))

        # 使用线性层进行维度变换，并应用批归一化
        DLM_cat_i = self.fc2_DLM(DLM_cat)
        DLM_cat_i = self.batchnorm2_DLM(DLM_cat_i)

        # 对染色质相关数据进行卷积处理
        chr_embedded = DLM_fea1.permute(0, 1, 2)  # 确保形状为[batch_size, in_channels, sequence_length]
        chr_conved = [self.Mish2(conv(chr_embedded)) for conv in self.convs_chr]
        chr_pooled = [
            F.max_pool1d(conv, math.ceil(conv.shape[2] // 10))
            for conv in chr_conved
        ]
        chr_flatten = [
            pool.contiguous().view(pool.size(0), -1)
            for pool in chr_pooled
        ]
        chr_cat = torch.cat(chr_flatten, dim=1)  # 将各分支连接在一起
        chr_cat = self.batchnorm_chr(chr_cat)  # 批归一化

        # 使用独立的全连接层fc2_chr
        chr_cat_i = self.fc2_chr(chr_cat)
        chr_cat_i = self.batchnorm2_chr(chr_cat_i)

        # 将序列数据和染色质相关数据融合在一起
        fusion, att_weight = self.ban1(DLM_cat_i.unsqueeze(1), chr_cat_i.unsqueeze(1))

        # 输出特征并分类
        return self.fc3(fusion), fusion  # 返回最终分类结果和中间特征


class TextCNN_block7(nn.Module):
    def __init__(self, vocab_size, embedding_dim_DLM, embedding_dim_seq, DLM_seq_len, sequence_length, n_filters,
                 filter_sizes, output_dim, dropout):
        super(TextCNN_block7, self).__init__()

        # 定义全连接层，用于将染色质特征从 919 维映射到 512 维
        self.fc1 = nn.Linear(919, 512)

        # 定义后续的特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(512, 256),  # 将 512 维特征映射到 256 维
            nn.Mish(),  # Mish 激活函数
            nn.Dropout(dropout),  # Dropout 层，防止过拟合
            nn.Linear(256, 256),  # 再次映射到 256 维，保持输出维度一致
            nn.Mish(),  # Mish 激活函数
        )

        # 定义分类层
        self.fc_classifier = nn.Linear(256, output_dim)  # 最终分类层，将 256 维特征映射到 output_dim

        # 定义批归一化层
        self.batchnorm1 = nn.BatchNorm1d(919)  # 对染色质数据进行批归一化
        self.batchnorm2 = nn.BatchNorm1d(512)  # 对隐藏层特征进行批归一化
        self.batchnorm3 = nn.BatchNorm1d(256)  # 对最终特征进行批归一化

    def forward(self, DLM_fea, seq_data, chr_data):
        # 对染色质相关数据进行批归一化
        fea_data = self.batchnorm1(chr_data)

        # 使用线性层进行维度变换
        fea = self.fc1(fea_data)
        fea = self.batchnorm2(fea)  # 批归一化
        fea = nn.Mish()(fea)  # 激活函数

        # 提取特征并降维到 256 维
        fea = self.feature_extractor(fea)
        fea = self.batchnorm3(fea)  # 批归一化

        # 输出分类结果
        final_output = self.fc_classifier(fea)

        # 返回最终分类结果和 256 维中间特征
        return final_output, fea  # 返回最终分类结果和中间特征


class TextCNN_block8(nn.Module):
    def __init__(self, vocab_size, embedding_dim_DLM, embedding_dim_seq, DLM_seq_len, sequence_length, n_filters,
                 filter_sizes, output_dim, dropout):
        super(TextCNN_block8, self).__init__()
        # 初始化多分枝卷积层
        self.convs1 = nn.ModuleList([nn.Conv1d(in_channels=12,  # 输入通道数
                                               out_channels=n_filters,  # 输出通道数
                                               kernel_size=fs,  # 卷积核大小
                                               padding='same')  # 使用'same'填充以保持输出长度与输入相同
                                     for fs in filter_sizes])  # 对每个filter size创建一个卷积层
        # 定义全连接层
        self.fc1 = nn.Linear(1920, 128)  # 将卷积后的特征映射到512维
        self.fc = nn.Sequential(
            nn.Linear(128, 32),  # 线性层，将512维特征映射到32维
            nn.Mish(),  # Mish激活函数
            nn.Dropout(),  # Dropout层，防止过拟合
            nn.Linear(32, 8),  # 线性层，将32维特征映射到8维
            nn.Mish(),  # Mish激活函数
            nn.Linear(8, output_dim)  # 最终输出层，将8维特征映射到output_dim
        )
        # 定义Dropout层和激活函数
        self.dropout1 = nn.Dropout(dropout)
        self.Mish1 = nn.Mish()
        # 定义批归一化层
        self.batchnorm1 = nn.BatchNorm1d(128)

    def forward(self, DLM_fea, seq_data, chr_data):
        # 对DLM特征进行维度变换，从[batch_size, sequence_length, embedding_dim]变为[batch_size, embedding_dim, sequence_length]
        DLM_embedded = DLM_fea.permute(0, 1, 2)
        # 应用卷积层并使用Mish激活函数
        DLM_conved = [self.Mish1(conv(DLM_embedded)) for conv in self.convs1]
        # 池化层，使用最大池化
        DLM_pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in DLM_conved]
        # 多分支线性展开
        DLM_flatten = [pool.contiguous().view(pool.size(0), -1) for pool in DLM_pooled]
        # 将各分支连接在一起
        DLM_cat = self.dropout1(torch.cat(DLM_flatten, dim=1))
        # 使用线性层进行维度变换，并应用批归一化
        DLM_cat_i = self.fc1(DLM_cat)
        DLM_cat_i = self.batchnorm1(DLM_cat_i)

        # 输出特征并分类
        return self.fc(DLM_cat_i), DLM_cat_i  # 返回最终分类结果和中间特征


class CombinedTextCNNBlock(nn.Module):
    def __init__(self, vocab_size, embedding_dim_DLM1, embedding_dim_DLM2, embedding_dim_seq, DLM_seq_len,
                 sequence_length, n_filters,
                 filter_sizes, output_dim, dropout):
        super(CombinedTextCNNBlock, self).__init__()

        # 初始化 TextCNN_block1 和 TextCNN_block3
        self.block1 = TextCNN_block1(vocab_size, embedding_dim_DLM1, embedding_dim_seq, DLM_seq_len,
                                     sequence_length, n_filters, filter_sizes, 256, dropout)
        self.block3 = TextCNN_block3(vocab_size, embedding_dim_DLM2, embedding_dim_seq, DLM_seq_len,
                                     sequence_length, n_filters, filter_sizes, 256, dropout)

        # 定义线性变换层，确保 block1 和 block3 的输出特征维度一致（假设为 256）
        self.fc_block1 = nn.Linear(256, 256)
        self.fc_block3 = nn.Linear(256, 256)

        # 定义 BAN 层用于融合 block1 和 block3 的特征
        self.ban = BANLayer(256, 256, 512, 2, 0.1, 6)

        # 定义最终分类层
        self.fc_final = nn.Sequential(
            nn.Linear(512, 32),  # 线性层，将256维特征映射到32维
            nn.Mish(),  # Mish激活函数
            nn.Dropout(dropout),  # Dropout层，防止过拟合
            nn.Linear(32, 8),  # 线性层，将32维特征映射到8维
            nn.Mish(),  # Mish激活函数
            nn.Linear(8, output_dim)  # 最终输出层，将8维特征映射到output_dim
        )

    def forward(self, DLM_fea, seq_data, chr_data):
        # 获取 block1 的输出特征
        block1_output, _ = self.block1(DLM_fea, seq_data, chr_data)
        block1_output = self.fc_block1(block1_output)

        # 获取 block3 的输出特征
        block3_output, _ = self.block3(DLM_fea, seq_data, chr_data)
        block3_output = self.fc_block3(block3_output)

        # 使用 BAN 层融合 block1 和 block3 的特征
        fusion, att_weight = self.ban(block1_output.unsqueeze(1), block3_output.unsqueeze(1))

        # 输出最终分类结果和中间特征
        final_output = self.fc_final(fusion.squeeze())

        return final_output, fusion.squeeze(), block1_output, block3_output


# 示例用法：
# combined_model = CombinedTextCNNBlock(vocab_size, embedding_dim_DLM, embedding_dim_seq, DLM_seq_len,
#                                       sequence_length, n_filters, filter_sizes, output_dim, dropout)

class MSCPred(nn.Module):
    def __init__(self, vocab_size, embedding_dim_DLM1, embedding_dim_DLM2, embedding_dim_seq, DLM_seq_len,
                 sequence_length, n_filters,
                 filter_size1, output_dim, dropout1, n_filters2, filter_size2, dropout2, filter_size3, dropout3,
                 filter_size4, dropout4):
        super(MSCPred, self).__init__()

        # 初始化两个编码器
        self.DLM_encoder1 = TextCNN_block1(vocab_size, embedding_dim_DLM1, embedding_dim_seq, DLM_seq_len,
                                           sequence_length, n_filters, filter_size1, output_dim, dropout1)
        self.DLM_encoder2 = TextCNN_block3(vocab_size, embedding_dim_DLM2, embedding_dim_seq, DLM_seq_len,
                                           sequence_length, n_filters2, filter_size2, output_dim, dropout2)
        self.DLM_encoder3 = TextCNN_block5(vocab_size, embedding_dim_DLM1, embedding_dim_seq, DLM_seq_len,
                                           sequence_length, n_filters2, filter_size3, output_dim, dropout3)
        self.DLM_encoder4 = TextCNN_block7(vocab_size, embedding_dim_DLM1, embedding_dim_seq, DLM_seq_len,
                                           sequence_length, n_filters2, filter_size2, output_dim, dropout2)
        self.DLM_encoder5 = TextCNN_block8(vocab_size, embedding_dim_DLM1, embedding_dim_seq, DLM_seq_len,
                                           sequence_length, n_filters2, filter_size2, output_dim, dropout2)

        self.DLM_encoder6 = TextCNN_block10(vocab_size, embedding_dim_DLM2, embedding_dim_seq, DLM_seq_len,
                                            sequence_length, n_filters2, filter_size4, output_dim, dropout4)

        self.seq_encoder1 = TextCNN_block2(vocab_size, embedding_dim_DLM1, embedding_dim_seq, DLM_seq_len,
                                           sequence_length, n_filters, filter_size1, output_dim, dropout1)
        self.seq_encoder2 = TextCNN_block4(vocab_size, embedding_dim_DLM2, embedding_dim_seq, DLM_seq_len,
                                           sequence_length, n_filters2, filter_size4, output_dim, dropout4)
        # self.seq_encoder3 = TextCNN_block6(vocab_size, embedding_dim_DLM2, embedding_dim_seq, DLM_seq_len,sequence_length, n_filters2, filter_sizes2, output_dim, dropout2)

        # 定义全连接层
        self.fc5 = nn.Sequential(
            nn.Linear(1024, 256),  # 线性层，将1024维特征映射到256维
            nn.Mish(),  # Mish激活函数
            nn.Dropout(),  # Dropout层，防止过拟合
            nn.Linear(256, 64),  # 线性层，将256维特征映射到64维
            nn.Mish(),  # Mish激活函数
            nn.Linear(64, output_dim)  # 最终输出层，将64维特征映射到output_dim
        )

        # 定义批归一化层
        self.batchnorm3 = nn.BatchNorm1d(1024)

    def forward(self, DLM_fea1, DLM_fea2, DLM_fea3, DLM_fea4, seq_data, chr_data):

        # 通过第一个编码器获取特征
        _, data1 = self.DLM_encoder1(DLM_fea1, seq_data, chr_data)
        # _, data2 = self.seq_encoder1(DLM_fea1, seq_data, chr_data)
        _, data3 = self.DLM_encoder2(DLM_fea2, seq_data, chr_data)
        _, data4 = self.seq_encoder2(DLM_fea4, seq_data, chr_data)
        _, data5 = self.DLM_encoder3(DLM_fea3, seq_data, chr_data)
        # _, data6 = self.DLM_encoder6(DLM_fea4, seq_data, chr_data)
        # _, data6 = self.seq_encoder3(DLM_fea3, seq_data, DLM_fea4)
        # _, data7 = self.seq_encoder3(DLM_fea2, seq_data, DLM_fea1)  # hyena突变序列及表观遗传修饰的融合特征(ban)
        # _, data8 = self.DLM_encoder4(DLM_fea3, seq_data, chr_data)
        #_, data9 = self.DLM_encoder5(DLM_fea4, seq_data, chr_data)

        # 将两类输出特征拼接
        fea = torch.cat([data1, data3, data5, data4], dim=1)
        fea = self.batchnorm3(fea)

        # 输出特征并分类
        return self.fc5(fea), fea  # 返回最终分类结果和中间特征