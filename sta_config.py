#!/usr/bin/env python
# coding=gbk

import argparse


def get_config():
    parse = argparse.ArgumentParser(description='default config')
    parse.add_argument('-a', type=str, default='model')
    # 数据参数
    parse.add_argument('-vocab_size', type=int, default=4, help='The size of the vocabulary')  # 词汇表大小
    parse.add_argument('-output_size', type=int, default=1, help='Number of mutation functions')  # 输出大小
    parse.add_argument('-CV', type=bool, default=False, help='Cross validation')  # 是否启用交叉验证
    # parse.add_argument('-CV', type=bool, default=True, help='Cross validation')  # 是否启用交叉验证

    # 训练参数
    parse.add_argument('-batch_size', type=int, default=128, help='Batch size')  # 批量大小，默认为 128。
    parse.add_argument('-epochs', type=int, default=200)  # 训练轮数，默认为 200。
    parse.add_argument('-learning_rate', type=float, default=0.0001)  # 学习率，默认为 0.0001。
    parse.add_argument('-threshold', type=float, default=0.5)  # 阈值，默认为 0.5。
    parse.add_argument('-early_stop', type=int, default=10)  # 早停机制的轮数，默认为 10。

    # 模型参数
    parse.add_argument('-model_name', type=str, default='MSCPred', help='Name of the model')
    parse.add_argument('-embedding_size_DLM1', type=int, default=768,
                       help='Dimension of the embedding')  # DNA语言模型特征维度，默认为 768。
    parse.add_argument('-embedding_size_DLM2', type=int, default=128,
                       help='Dimension of the embedding')  # DNA语言模型特征维度，默认为 768。
    parse.add_argument('-DLM_seq_len', type=int, default=128,
                       help='Length of the sequence in DLM model')  # DNA语言模型序列长度，默认为 128。
    parse.add_argument('-embedding_size_seq', type=int, default=128,
                       help='Dimension of the embedding')  # 序列相关的特征维度，默认为 128。
    parse.add_argument('-sequence_length', type=int, default=1001,
                       help='Length of the mutation sequence')  # 突变序列长度，默认为 1001。

    parse.add_argument('-dropout1', type=float, default=0.5)  # Dropout 概率
    parse.add_argument('-filter_num', type=int, default=64, help='Number of the filter')  # 卷积核数量，默认为 64。
    parse.add_argument('-filter_size1', type=list, default=[1, 4, 7], help='Size of the filter')

    parse.add_argument('-dropout2', type=float, default=0.4)  # Dropout 概率
    parse.add_argument('-filter_num2', type=int, default=64, help='Number of the filter')  # 卷积核数量，默认为 64。
    parse.add_argument('-filter_size2', type=list, default=[2, 4, 8], help='Size of the filter')

    parse.add_argument('-dropout3', type=float, default=0.8)  # Dropout 概率
    parse.add_argument('-filter_size3', type=list, default=[2, 4, 6], help='Size of the filter')

    parse.add_argument('-dropout4', type=float, default=0.8)  # Dropout 概率
    parse.add_argument('-filter_size4', type=list, default=[2, 3, 6], help='Size of the filter')

    # 路径参数
    ## 训练集参考序列特征文件路径，默认为 ./data/train_GPN-MSA_feature.pth
    parse.add_argument('-train_direction1', type=str, default='./data/train_GPN-MSA_feature.pth',help='The ref-seq feature of training set')
    parse.add_argument('-train_direction2', type=str, default='./data/train_Hyena_feature-ref.pth',help='The ref-seq feature of training set')
    parse.add_argument('-train_direction3', type=str, default='./data/train_calm.pth',help='The ref-seq feature of training set')
    parse.add_argument('-train_direction4', type=str, default='./data/train-alt-rna.npy',help='The ref-seq feature of training set')
    ## 训练集突变序列文件路径，默认为 ./data/train.csv。
    parse.add_argument('-train_label_direction', type=str, default='./data/train.csv',help='The Mut-seq of training set')
    ## 训练集表观遗传特征文件路径，默认为 ./data/train_DanQ_feature.h5。
    parse.add_argument('-chrom_train_direction', type=str, default='./data/train_DanQ_feature.h5',help='The epigenetic feature of training set')
    ## 测试集参考序列特征文件路径，默认为 ./data/test_GPN-MSA_feature.pth。
    parse.add_argument('-test_direction1', type=str, default='./data/test_GPN-MSA_feature.pth',help='The ref-seq feature of test set')
    parse.add_argument('-test_direction2', type=str, default='./data/test_Hyena_feature-ref.pth',help='The ref-seq feature of test set')
    parse.add_argument('-test_direction3', type=str, default='./data/test_calm.pth',help='The ref-seq feature of training set')
    parse.add_argument('-test_direction4', type=str, default='./data/test-alt-rna.npy',help='The ref-seq feature of training set')
    ## 测试集突变序列文件路径，默认为 ./data/test.csv。
    parse.add_argument('-test_label_direction', type=str, default='./data/test.csv', help='The Mut-seq of test set')
    ## 测试集表观遗传特征文件路径，默认为 ./data/test_DanQ_feature.h5。
    parse.add_argument('-chrom_test_direction', type=str, default='./data/test_DanQ_feature.h5',help='The epigenetic feature of test set')

    config = parse.parse_args()
    return config

