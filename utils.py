import os
from datetime import datetime
import pandas as pd
import json
import yaml
import numpy as np
import torch
from sklearn.metrics import accuracy_score

def weight_freeze(model) :
    for i, child in enumerate(model.children()) :
        for n, p in child.named_modules() :
            if n != 'classifier' :
                for param in p.parameters():
                    param.requires_grad = False
            elif n == 'classifier' :
                for param in p.parameters():
                    param.requires_grad = True
    return model


def weight_load(model, optimizer, ckpt, training=True):
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])

    if training :
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer, checkpoint['epoch']

    else :
        return model

def get_models(model, checkpoint):
    models = []
    for path in checkpoint:
        models.append(weight_load(model, None, path, training=False))
        print(f"MODEL LOAD ... from {path}")

    if len(checkpoint) == 0:
        return models[0]
    else:
        return models

def save_to_csv(df, preds, save_path):
    df['label'] = preds
    df.to_csv(save_path, index=False)

def save_config(cfg, output) :
    os.makedirs(output, exist_ok=True)
    cfg_save_name = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    with open(os.path.join(output, f"{cfg_save_name}.json"), 'w') as f:
        json.dump(cfg, f, indent="\t")

def logging(logger, data, step) :
    for i, (k,v) in enumerate(data.items()) :
        logger.add_scalar(k, v, step)


def score(true_labels, model_preds) :
    model_preds = model_preds.argmax(1).detach().cpu().numpy().tolist()
    true_labels = true_labels.detach().cpu().numpy().tolist()
    return accuracy_score(true_labels, model_preds)

def tensor2list(x):
    return x.detach().cpu().numpy().tolist()

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(imgs, labels):
    lam = np.random.beta(1.0, 1.0)
    rand_index = torch.randperm(imgs.size()[0]).cuda()
    target_a = labels
    target_b = labels[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
    imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))

    return imgs, lam, target_a, target_b


def mixup(imgs, labels):
    lam = np.random.beta(1.0, 1.0)
    rand_index = torch.randperm(imgs.size()[0]).cuda()
    mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index, :]
    target_a, target_b = labels, labels[rand_index]

    return mixed_imgs, lam, target_a, target_b


def read_cfg(path) :
    with open(path, encoding='UTF-8') as f:
        y = yaml.load(f, Loader=yaml.FullLoader)
    return y


class LABEL_ENCODER() :
    def __init__(self, path):
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

class CATEGORY_CLS_ENCODER() :
    def __init__(self, path='./data/train.csv', device="cuda"):
        self.base_encoder = LABEL_ENCODER(path=path).base_encoder
        self.cat1_encoder = self.cat1_cls_encoder()
        self.cat2_encoder = self.cat2_cls_encoder()
        self.cat3_encoder = self.cat3_cls_encoder()
        self.device = device
    def __call__(self, prev, cur, category):
        prev = tensor2list(prev) if prev is not None else None
        cur = tensor2list(cur)

        if category == 1:
            cls_num = [self.cat1_encoder[c] for c in  cur]
            return cls_num

        elif category == 2:
            cls_num = torch.tensor([self.cat2_encoder[p][c] for p, c in zip(prev, cur)])
            return cls_num.to(self.device)

        elif category == 3:
            cls_num = torch.tensor([self.cat3_encoder[p][c] for p, c in zip(prev, cur)])
            return cls_num.to(self.device)

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