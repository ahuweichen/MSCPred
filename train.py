#!/usr/bin/env python
# coding=gbk

import time
import torch
import math
import numpy as np
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
import os

class DataTrain:
    def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = scheduler
        self.device = device

    def train_step(self, train_iter, test_iter, model_name, epochs=None, threshold=0.5):
        steps = 1
        train_fea = []
        best_loss = 100000.
        best_loss_acc = 0.
        bestlos_epoch = 0
        PATH = os.getcwd()
        best_model = os.path.join(PATH, 'result', 'best.pth')
        early_stop = 10

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0
            alpha = 0.4
            i = 0

            for train_data1, train_data2, train_data3, train_data4, seq_data, chr_data, train_label in train_iter:
                self.model.train()
                train_data1, train_data2, train_data3, train_data4, seq_data, chr_data, train_label = (
                    train_data1.to(self.device), train_data2.to(self.device), train_data3.to(self.device),
                    train_data4.to(self.device), seq_data.to(self.device), chr_data.to(self.device),
                    train_label.to(self.device)
                )

                # 前向传播
                y_hat, train_feature = self.model(train_data1, train_data2, train_data3, train_data4, seq_data, chr_data)
                loss = self.criterion(y_hat, train_label.float().unsqueeze(1))

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 更新学习率
                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        self.lr_scheduler.step()
                    else:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                total_loss += loss.item()
                steps += 1

            end_time = time.time()
            epoch_time = end_time - start_time

            # 在训练集上进行预测
            model_predictions, true_labels = predict(self.model, train_iter, device=self.device)
            for i in range(len(model_predictions)):
                if model_predictions[i] < threshold:
                    model_predictions[i] = 0
                else:
                    model_predictions[i] = 1
            y_hat = model_predictions
            acc1 = accuracy_score(true_labels, y_hat)

            print(f'{model_name}|Epoch:{epoch:003} | Time:{epoch_time:.2f}s')
            print(f'Train loss:{total_loss / len(train_iter)}')
            print(f'Train acc:{acc1}')

            train_loss = total_loss / len(train_iter)
            if train_loss < best_loss:
                torch.save(self.model.state_dict(), best_model)
                best_loss = train_loss
                best_loss_acc = acc1
                bestlos_epoch = epoch

            if (best_loss < train_loss) and (epoch - bestlos_epoch >= early_stop):
                break

        self.model.load_state_dict(torch.load(best_model))
        print("best_loss = " + str(best_loss))
        print("best_loss_acc = " + str(best_loss_acc))


def predict(model, data, device="cuda"):
    # 模锟斤拷预锟斤拷
    model.to(device)  # 锟斤拷模锟斤拷锟狡讹拷锟斤拷指锟斤拷锟借备
    model.eval()  # 锟斤拷锟斤拷锟斤拷锟斤拷模式
    predictions = []  # 锟芥储预锟斤拷锟斤拷
    labels = []  # 锟芥储锟斤拷锟斤拷实锟斤拷签

    with torch.no_grad():  # 取锟斤拷锟捷度凤拷锟津传诧拷
        for x, x2, x3,x4,f, f2, y in data:
            x = x.to(device)  # 锟斤拷锟斤拷锟斤拷锟狡讹拷锟斤拷指锟斤拷锟借备
            x2 = x2.to(device)
            x3 = x3.to(device)
            x4 = x4.to(device)
            f = f.to(device)
            f2 = f2.to(device)
            y = y.to(device).unsqueeze(1)  # 锟斤拷实锟斤拷签

            score, _ = model(x, x2, x3, x4, f, f2)  # 模锟斤拷预锟斤拷
            label = torch.sigmoid(score)  # 锟斤拷模锟斤拷预锟斤拷值映锟斤拷锟斤拷0-1之锟斤拷
            predictions.extend(label.tolist())  # 锟斤拷预锟斤拷锟斤拷锟斤拷锟接碉拷锟叫憋拷
            labels.extend(y.tolist())  # 锟斤拷锟斤拷实锟斤拷签锟斤拷锟接碉拷锟叫憋拷

    return np.array(predictions), np.array(labels)  # 锟斤拷锟斤拷预锟斤拷锟斤拷锟斤拷锟斤拷实锟斤拷签

# def predict(model, data, device="cuda"):
#     model.to(device)
#     model.eval()
#     predictions = []
#     labels = []
#
#     with torch.no_grad():
#         for x, x2, x3, x4, f, f2, y in data:
#             x = x.to(device)
#             x2 = x2.to(device)
#             x3 = x3.to(device)
#             x4 = x4.to(device)
#             f = f.to(device)
#             f2 = f2.to(device)
#             y = y.to(device).unsqueeze(1)
#
#             score, _ = model(x, x2, x3, x4, f, f2)
#             label = torch.sigmoid(score)
#             predictions.extend(label.tolist())
#             labels.extend(y.tolist())
#
#     return np.array(predictions), np.array(labels)

def get_linear_schedule_with_warmup(optimizer_, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer_, lr_lambda, last_epoch)


class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch - 1) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                           (1 + math.cos(math.pi * (epoch - 1 - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr


def feature(model, dataloader, device="cuda"):
    """
    提取模型的中间特征表示
    :param model: 训练好的模型
    :param dataloader: 数据加载器
    :param device: 运行设备
    :return: 所有样本的特征张量 (N, feature_dim)
    """
    model.eval()  # 设置为评估模式
    features_list = []

    with torch.no_grad():  # 不计算梯度
        for batch in dataloader:
            # 假设 batch 是 (x, x2, x3, x4, f, f2, y)
            x, x2, x3, x4, f, f2, _ = batch  # 最后一个 y 是标签，我们不需要

            # 移动到指定设备
            x = x.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            x4 = x4.to(device)
            f = f.to(device)
            f2 = f2.to(device)

            # 前向传播，获取特征
            # 注意：MSCPred.forward 返回 (logits, features)
            logits, feats = model(x, x2, x3, x4, f, f2)

            # 将特征添加到列表
            features_list.append(feats.cpu()) # 移动到 CPU 并添加到列表

    # 拼接所有 batch 的特征
    all_features = torch.cat(features_list, dim=0)
    return all_features



