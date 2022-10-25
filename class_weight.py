import os
from datetime import datetime
from glob import glob
import pandas as pd
import json
import yaml
import numpy as np
import torch
from sklearn.metrics import accuracy_score

class LABEL_ENCODER() :
    def __init__(self, path='./data/train.csv'):
        self.base_encoder = self.base_label_encoder(path)

    def base_label_encoder(self, path='./data/train.csv') :
        df = pd.read_csv(path)
        cat_encoder = {n_cat1: {} for n_cat1 in df['cat1'].unique()}
        for n_cat1 in df['cat1'].unique():
            cat2 = df[df['cat1'] == n_cat1]['cat2'].unique()

            for n_cat2 in cat2:
                cat3 = df[df['cat1'] == n_cat1][df['cat2'] == n_cat2]['cat3'].unique()
                cat_encoder[n_cat1][n_cat2] = []

                for n_cat3 in cat3:
                    cat_encoder[n_cat1][n_cat2].append(n_cat3)
        return cat_encoder

    def cat1_label_index_encoder(self) :
        cat1_encoder = {k: i for i, k in enumerate(self.base_encoder.keys())}
        return cat1_encoder

    def cat1_label_encoder(self):
        return self.cat1_label_index_encoder()

    def cat2_label_index_encoder(self) :
        cat2_encoder = []
        cnt = 0
        for cat1_k in self.base_encoder.keys():
            enc = {}
            for cat2_k in self.base_encoder[cat1_k].keys() :
                enc[cat2_k] = cnt
                cnt += 1
            cat2_encoder.append(enc)
        return cat2_encoder

    def cat2_label_encoder(self):
        cat2_encoder = {}
        cnt = 0
        for cat1_k in self.base_encoder.keys():
            for cat2_k in self.base_encoder[cat1_k].keys() :
                cat2_encoder[cat2_k] = cnt
                cnt += 1
        return cat2_encoder

    def cat3_label_index_encoder(self) :
        cat3_encoder = []
        cnt = 0
        for cat1_k in self.base_encoder.keys():
            for cat2_k in self.base_encoder[cat1_k].keys():
                enc = {}
                for cat3_k in self.base_encoder[cat1_k][cat2_k]:
                    enc[cat3_k] = cnt
                    cnt += 1
                cat3_encoder.append(enc)
        return cat3_encoder

    def cat3_label_encoder(self) :
        cat3_encoder = {}
        cnt = 0
        for cat1_k in self.base_encoder.keys():
            for cat2_k in self.base_encoder[cat1_k].keys():
                for cat3_k in self.base_encoder[cat1_k][cat2_k]:
                    cat3_encoder[cat3_k] = cnt
                    cnt += 1
        return cat3_encoder

class CATEGORY_MASKING() :
    def __init__(self, path='./data/train.csv', device="cuda"):
        self.base_encoder = LABEL_ENCODER(path=path).base_encoder
        self.cat1_encoder = self.cat1_cls_encoder()
        self.cat2_encoder = self.cat2_cls_encoder()
        self.cat3_encoder = self.cat3_cls_encoder()
        self.device = device

    def __call__(self, pred, category):
        # prev = tensor2list(prev) if prev is not None else None
        batch_size = pred.shape[0]
        pred = tensor2list(pred)

        if category == 1:
            mask = torch.tensor([[0] * 18] * batch_size, dtype=torch.float32)
            for idx, c in enumerate(pred) :
                mask[idx][list(self.cat2_encoder[c].values())] = 1
            return mask

        elif category == 2:
            mask = torch.tensor([[0] * 128] * batch_size, dtype=torch.float32)
            for idx, c in enumerate(pred) :
                mask[idx][list(self.cat3_encoder[c].values())] = 1
            return mask

        # elif category == 3:
        #     mask = [[True] * 128] * cur.shape[0]
        #     for idx, (p, c) in enumerate(zip(prev, cur)):
        #         mask[idx][self.cat3_encoder[p][c]] = False
        #     cls_num = torch.tensor([self.cat3_encoder[p][c] for p, c in zip(prev, cur)])
        #     return cls_num.to(self.device)

    def cat1_cls_encoder(self) :
        # cat1_encoder = {0:0, 1:1, 2:5, 3:2, 4:14, 5:15}
        cat1_encoder = [0, 1, 2, 3, 4, 5]
        return cat1_encoder

    def cat2_cls_encoder(self) :
        cat2_encoder = []
        cnt = 0
        for cat1_k in self.base_encoder.keys():
            cat2 = {}
            for i, cat2_k in enumerate(self.base_encoder[cat1_k].keys()) :
                cat2[i] = cnt
                cnt += 1
            cat2_encoder.append(cat2)
        return cat2_encoder

    def cat3_cls_encoder(self) :
        cat3_encoder = []
        cnt = 0
        for cat1_k in self.base_encoder.keys():
            for cat2_k in self.base_encoder[cat1_k].keys():
                cat3 = {}
                for i, cat3_k in enumerate(self.base_encoder[cat1_k][cat2_k]) :
                    cat3[i] = cnt
                    cnt+=1
                cat3_encoder.append(cat3)
        return cat3_encoder

class CLS_WEIGHT() :
    def __init__(self, csv_path='./data/train.csv', device="cuda"):
        self.masking_generator = CATEGORY_MASKING(path=csv_path)
        self.device = device

    def __call__(self, pred, num, category):
        batch = pred.shape[0]
        prev_cls_num = pred.shape[1]

        pred_sm = pred.softmax(dim=1)
        mask2 = torch.zeros(1, num).to(self.device)
        for i in range(prev_cls_num):
            mask3 = torch.zeros(1).to(self.device)
            mask = self.masking_generator(torch.tensor([i] * batch), category=category).to(self.device)
            for j in range(batch):
                # mask[j, :] = torch.mul(mask[j, :],  pred_sm[j, i])
                mask3 = torch.cat([mask3, torch.mul(mask[j, :], pred_sm[j, i])], dim=0)
            mask2 = torch.cat([mask2, mask3[1:].reshape(batch, num)], dim=0)
        return torch.sum(mask2[1:].reshape(prev_cls_num, batch, num), dim=0)