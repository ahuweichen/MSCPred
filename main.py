#!/usr/bin/env python
# coding=gbk

import csv
import os
import time
import numpy as np
import pandas as pd
import torch
import h5py
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

import estimate
import sta_config
from models.MSCPred import MSCPred
from train import DataTrain, predict, CosineScheduler, feature

torch.manual_seed(20250423)
torch.backends.cudnn.deterministic = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bases = 'ATCG'


def one_hot_encode_dna(sequence, bases='ATCG', seq_length=1001):
    for seq in sequence:
        if len(seq) != seq_length:
            raise ValueError(f"序列长度错误，所有序列的长度必须等于{seq_length}")

    one_hot = np.zeros((len(sequence), seq_length, len(bases)), dtype=np.float32)

    for i, seq in enumerate(sequence):
        for j, base in enumerate(seq):
            if base in bases:
                one_hot[i, j, bases.index(base)] = 1
            else:
                pass

    return one_hot


def getSequenceData(direction1, direction2, direction3, direction4, chrom_direction, label_direction):
    if direction1.endswith('.pth'):
        data1 = torch.load(direction1)
    elif direction1.endswith('.npy'):
        data1 = torch.from_numpy(np.load(direction1)).float()
    else:
        raise ValueError(f"Unsupported file format: {direction1}")

    if direction2.endswith('.pth'):
        data2 = torch.load(direction2)
    elif direction2.endswith('.npy'):
        data2 = torch.from_numpy(np.load(direction2)).float()
    else:
        raise ValueError(f"Unsupported file format: {direction2}")

    if direction3.endswith('.pth'):
        data3 = torch.load(direction3)
    elif direction3.endswith('.npy'):
        data3 = torch.from_numpy(np.load(direction3)).float()
    else:
        raise ValueError(f"Unsupported file format: {direction3}")

    if direction4.endswith('.pth'):
        data4 = torch.load(direction4)
    elif direction4.endswith('.npy'):
        data4 = torch.from_numpy(np.load(direction4)).float()
    else:
        raise ValueError(f"Unsupported file format: {direction4}")

    Frame = pd.read_csv(label_direction)
    sequence = Frame["ALT_seq"].values
    label = torch.tensor(Frame["Label"].values, dtype=torch.long)

    one_hot_sequences = one_hot_encode_dna(sequence)
    one_hot_sequences_tensor = torch.tensor(one_hot_sequences, dtype=torch.float32)

    chrom = h5py.File(chrom_direction, 'r')
    alt = np.array(chrom['feat_alt'])
    chrom_fea = torch.tensor(alt)

    return data1, data2, data3, data4, one_hot_sequences_tensor, chrom_fea, label


def data_load(train_direction1, train_direction2, train_direction3, train_direction4, chrom_train_direction, train_label_direction, test_direction1, test_direction2, test_direction3, test_direction4, chrom_test_direction, test_label_direction, batch, encode='embedding', cv=True, SH=True):
    dataset_train, dataset_test = [], []
    dataset_va = None
    assert encode in ['embedding', 'sequence'], 'There is no such representation!!!'

    if cv:
        dataset_va = []
        dataset_train = []

        x_train, x_train1, x_train2, x_train3, seq_train, chr_train, y_train = getSequenceData(
            train_direction1, train_direction2, train_direction3, train_direction4, chrom_train_direction, train_label_direction
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=6)
        for i, (train_index, test_index) in enumerate(cv.split(x_train, y_train)):
            data_train, data_train1, data_train2, data_train3, sequence_train, chrom_train, label_train = (
                x_train[train_index], x_train1[train_index], x_train2[train_index], x_train3[train_index],
                seq_train[train_index], chr_train[train_index], y_train[train_index]
            )
            data_test, data_test1, data_test2, data_test3, sequence_test, chrom_test, label_test = (
                x_train[test_index], x_train1[test_index], x_train2[test_index], x_train3[test_index],
                seq_train[test_index], chr_train[test_index], y_train[test_index]
            )

            train_data = TensorDataset(
                torch.tensor(data_train, dtype=torch.float32, requires_grad=False),
                torch.tensor(data_train1, dtype=torch.float32, requires_grad=False),
                torch.tensor(data_train2, dtype=torch.float32, requires_grad=False),
                torch.tensor(data_train3, dtype=torch.float32, requires_grad=False),
                torch.tensor(sequence_train, dtype=torch.float32, requires_grad=False),
                torch.tensor(chrom_train, dtype=torch.float32, requires_grad=False),
                torch.tensor(label_train, dtype=torch.long, requires_grad=False)
            )
            test_data = TensorDataset(
                torch.tensor(data_test, dtype=torch.float32, requires_grad=False),
                torch.tensor(data_test1, dtype=torch.float32, requires_grad=False),
                torch.tensor(data_test2, dtype=torch.float32, requires_grad=False),
                torch.tensor(data_test3, dtype=torch.float32, requires_grad=False),
                torch.tensor(sequence_test, dtype=torch.float32, requires_grad=False),
                torch.tensor(chrom_test, dtype=torch.float32, requires_grad=False),
                torch.tensor(label_test, dtype=torch.long, requires_grad=False)
            )

            dataset_train.append(DataLoader(train_data, batch_size=batch, shuffle=SH, num_workers=4))
            dataset_va.append(DataLoader(test_data, batch_size=batch, shuffle=SH, num_workers=4))

    else:
        print("encode train")
        x_train, x_train1, x_train2, x_train3, seq_train, chr_train, y_train = getSequenceData(
            train_direction1, train_direction2, train_direction3, train_direction4, chrom_train_direction, train_label_direction
        )
        train_data = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32, requires_grad=False),
            torch.tensor(x_train1, dtype=torch.float32, requires_grad=False),
            torch.tensor(x_train2, dtype=torch.float32, requires_grad=False),
            torch.tensor(x_train3, dtype=torch.float32, requires_grad=False),
            torch.tensor(seq_train, dtype=torch.float32, requires_grad=False),
            torch.tensor(chr_train, dtype=torch.float32, requires_grad=False),
            torch.tensor(y_train, dtype=torch.long, requires_grad=False)
        )
        dataset_train.append(DataLoader(train_data, batch_size=batch, shuffle=SH, num_workers=4))

    print("encode test")
    x_test, x_test1, x_test2, x_test3, seq_test, chr_test, y_test = getSequenceData(
        test_direction1, test_direction2, test_direction3, test_direction4, chrom_test_direction, test_label_direction
    )
    test_data = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32, requires_grad=False),
        torch.tensor(x_test1, dtype=torch.float32, requires_grad=False),
        torch.tensor(x_test2, dtype=torch.float32, requires_grad=False),
        torch.tensor(x_test3, dtype=torch.float32, requires_grad=False),
        torch.tensor(seq_test, dtype=torch.float32, requires_grad=False),
        torch.tensor(chr_test, dtype=torch.float32, requires_grad=False),
        torch.tensor(y_test, dtype=torch.long, requires_grad=False)
    )
    dataset_test.append(DataLoader(test_data, batch_size=batch, shuffle=False, num_workers=4))

    return dataset_train, dataset_va, dataset_test


def spent_time(start, end):
    epoch_time = end - start
    minute = int(epoch_time / 60)
    secs = int(epoch_time - minute * 60)
    return minute, secs


def save_results(model_name, data_loading_time, training_time, testing_time, test_score, file_path):
    title = ['Model', 'Recall', 'SPE', 'Precision', 'F1', 'MCC', 'Acc', 'AUC', 'AUPR',
             'DataLoadingTime', 'TrainingTime', 'TestingTime', 'RunTime', 'Test_Time']

    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    content = [[model_name,
                '%.3f' % test_score[0],
                '%.3f' % test_score[1],
                '%.3f' % test_score[2],
                '%.3f' % test_score[3],
                '%.3f' % test_score[4],
                '%.3f' % test_score[5],
                '%.3f' % test_score[6],
                '%.3f' % test_score[7],
                '%.3f' % data_loading_time,
                '%.3f' % training_time,
                '%.3f' % testing_time,
                '%.3f' % (data_loading_time + training_time + testing_time),
                now]]

    if os.path.exists(file_path):
        data = pd.read_csv(file_path, header=None, encoding='gbk')
        one_line = list(data.iloc[0])
        if one_line == title:
            with open(file_path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerows(content)
        else:
            with open(file_path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerow(title)
                writer.writerows(content)
    else:
        with open(file_path, 'a+', newline='') as t:
            writer = csv.writer(t)
            writer.writerow(title)
            writer.writerows(content)


def main(paths=None):
    print("doing: start-lost predition")

    Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # parse_file = f"./result/sta_pares.txt"
    # file1 = open(parse_file, 'a')
    # file1.write(Time)
    # file1.write('\n')
    # print(args, file=file1)
    # file1.write('\n')
    # file1.close()
    # file_path = "{}/2/{}.csv".format('result', 'sta_test')

    global_start_time = time.time()
    data_loading_start_time = time.time()

    print("Data is loading......")
    train_datasets, va_datasets, test_datasets = data_load(
        args.train_direction1, args.train_direction2, args.train_direction3, args.train_direction4, args.chrom_train_direction,
        args.train_label_direction, args.test_direction1, args.test_direction2, args.test_direction3, args.test_direction4,
        args.chrom_test_direction, args.test_label_direction,
        args.batch_size, cv=args.CV
    )
    data_loading_end_time = time.time()
    data_loading_time = data_loading_end_time - data_loading_start_time
    print(f"Data loading completed in {data_loading_time:.2f} seconds")

    all_test_score = 0

    if paths is None:
        print(f"{args.model_name} is training......")
        for i in range(len(train_datasets)):
            train_dataset = train_datasets[i]
            test_dataset = test_datasets[0]

            train_start_time = time.time()

            model = MSCPred(args.vocab_size, args.embedding_size_DLM1, args.embedding_size_DLM2, args.embedding_size_seq, args.DLM_seq_len,
                              args.sequence_length, args.filter_num, args.filter_size1, args.output_size, args.dropout1, args.filter_num2, args.filter_size2, args.dropout2, args.filter_size3, args.dropout3, args.filter_size4, args.dropout4)

            model_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # file2 = open(models_file, 'a')
            # file2.write(model_time)
            # file2.write('\n')
            # print(model, file=file2)
            # file2.write('\n')
            # file2.close()

            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            lr_scheduler = CosineScheduler(10000, base_lr=args.learning_rate, warmup_steps=500)
            criterion = torch.nn.BCEWithLogitsLoss()

            Train = DataTrain(model, optimizer, criterion, lr_scheduler, device=DEVICE)

            if va_datasets is None:
                Train.train_step(train_dataset, test_dataset, args.model_name, args.epochs,
                                 threshold=args.threshold)
            else:
                test_dataset = va_datasets[i]
                Train.train_step(train_dataset, va_datasets, args.model_name, args.epochs,
                                 threshold=args.threshold)

            training_end_time = time.time()
            training_time = training_end_time - train_start_time
            print(f"Training completed in {training_time:.2f} seconds")

            # testing_start_time = time.time()
            model_predictions, true_labels = predict(model, test_dataset, device=DEVICE)
            # testing_end_time = time.time()
            # testing_time = testing_end_time - testing_start_time
            # print(f"Testing completed in {testing_time:.2f} seconds")
            # result = pd.DataFrame(model_predictions)  ###输出预测分数
            # result.to_csv('./newresult/2/test_pred_score.txt', sep='\t', index=False, header=False)
            test_score = estimate.scores(model_predictions, true_labels, args.threshold)

            # save_results(args.model_name, data_loading_time, training_time, testing_time, test_score, file_path)

            print(f"{args.model_name}, test set:")
            metric = ["Recall", "SPE", "Precision", "F1", "MCC", "Acc", "AUC", "AUPR"]
            for k in range(len(metric)):
                print(f"{metric[k]}: {test_score[k]}\n")
            #             # 重新加载数据集并输出每个样本经深度学习模型编码后的特征
            print("Data is reloading......")
            train_datasets, va_datasets, test_datasets = data_load(args.train_direction1, args.train_direction2, args.train_direction3, args.train_direction4, args.chrom_train_direction,args.train_label_direction, args.test_direction1, args.test_direction2, args.test_direction3, args.test_direction4,args.chrom_test_direction, args.test_label_direction,args.batch_size, cv=args.CV)
            #train_dataset = train_datasets[0]
            train_fea = feature(model, train_datasets)
            # torch.save(train_fea, "./save_feature/1-training_TextCNN.h5")
            train_fea = pd.DataFrame(train_fea.cpu().detach().numpy()) ###保存为numpy数组
            train = pd.read_csv(args.train_label_direction)   ###重新加载包含标签的数据
            train_label = train["Label"] ##提取标签
            train_fea = pd.concat([train_label, train_fea], axis=1)   ##将样本特征和标签拼接在一起后输出，作为后续分类器的输入
            train_fea.to_csv('./save_feature/1-training_TextCNN.txt', sep='\t', index=False, header=False)

            #test_dataset = test_datasets[0]
            test_fea = feature(model, test_datasets)
            # torch.save(test_fea, "./save_feature/2-testing_TextCNN.h5")
            test_fea = pd.DataFrame(test_fea.cpu().detach().numpy())
            test = pd.read_csv(args.test_label_direction)
            test_label = test["Label"]
            test_fea = pd.concat([test_label, test_fea], axis=1)
            test_fea.to_csv('./save_feature/2-testing_TextCNN.txt', sep='\t', index=False, header=False)

    global_end_time = time.time()
    total_time = global_end_time - global_start_time
    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == '__main__':
    # models_file = f'./newresult/2/model_details.txt'
    args = sta_config.get_config()
    main()