# -*- coding: utf-8 -*-
import numpy as np
import sys
import csv
import torch
import logging
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate
csv.field_size_limit(sys.maxsize)


def get_dataset_nlp(dataset, data_path, batch_size=256):
    if dataset == 'NLP':
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
        input_dim = len(alphabet)
        max_length = 1014
        traindir = '/nfs/data/ruocwang/data/TextClassificationDatasets/dbpedia_csv/train.csv'
        validdir = '/nfs/data/ruocwang/data/TextClassificationDatasets/dbpedia_csv/test.csv'
        dst_train = SADataset(traindir, max_length)
        dst_test = SADataset(validdir, max_length)
        num_classes = 14
    else:
        exit('unknown dataset: %s'%dataset)
    trainloader_full = torch.utils.data.DataLoader(dst_train, batch_size=batch_size, shuffle=False, num_workers=2) # pin memory
    testloader_full = torch.utils.data.DataLoader(dst_test, batch_size=batch_size, shuffle=False, num_workers=2) # pin memory
    return max_length, input_dim, num_classes, dst_train, dst_test, trainloader_full, testloader_full


class SADataset(Dataset):
    def __init__(self, data_path, max_length=1014):
        self.data_path = data_path
        self.vocabulary = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
        self.identity_mat = np.identity(len(self.vocabulary))
        texts, labels = [], []
        with open(data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
                text = ""
                for tx in line[1:]:
                    text += tx
                    text += " "
                # if len(line) == 3:
                #     text = "{} {}".format(line[1].lower(), line[2].lower())
                # else:
                #     text = "{}".format(line[1].lower())
                label = int(line[0]) - 1
                texts.append(text)
                labels.append(label)
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.length = len(self.labels)
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.texts[index]
        data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text) if i in self.vocabulary],
                        dtype=np.float32)
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data), len(self.vocabulary)), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros((self.max_length, len(self.vocabulary)), dtype=np.float32)
        label = self.labels[index]
        data = torch.tensor(data)
        return data, label
    

class TensorDataset(Dataset):
    def __init__(self, texts, labels, transform=None): # images: n x c x h x w tensor
        self.texts = texts.detach().float()
        self.labels = labels.detach()
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            self.transform(self.texts[index])
        return self.texts[index], self.labels[index]

    def __len__(self):
        return self.texts.shape[0]

