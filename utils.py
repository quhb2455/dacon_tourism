import os
from datetime import datetime
from glob import glob
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

def single_model_weight_load(model, optimizer, ckpt, training=True):
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])

    if training :
        print(f"Weight Loaded From {ckpt}..")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer, checkpoint['epoch']

    else :
        return model


def weight_load(backbone, head1, head2, head3, optimizer, ckpt, training=True):
    checkpoint = torch.load(ckpt)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    head1.load_state_dict(checkpoint['head1_state_dict'])
    head2.load_state_dict(checkpoint['head2_state_dict'])
    head3.load_state_dict(checkpoint['head3_state_dict'])

    if training :
        print(f"Weight Loaded From {ckpt}..")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return backbone, head1, head2, head3, optimizer, checkpoint['epoch']

    else :
        return backbone, head1, head2, head3

def get_models(backbone, head1, head2, head3, checkpoint):
    backbones, head1s, head2s, head3s = [], [], [], []
    for path in checkpoint:
        _backbone, _head1, _head2, _head3 = weight_load(backbone, head1, head2, head3, None, path, training=False)
        backbones.append(_backbone)
        head1s.append(_head1)
        head2s.append(_head2)
        head3s.append(_head3)
        print(f"MODEL LOAD ... from {path}")

    if len(checkpoint) == 1:
        return backbones[0], head1s[0], head2s[0], head3s[0]
    else:
        return backbones, head1s, head2s, head3s


def save_to_csv(df, preds, save_path):
    df['cat3'] = preds
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

def list2tensor(x, dtype=torch.float, device="cuda"):
    return torch.tensor(x, dtype=dtype).to(device)

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


def read_cfg(path, specific=None) :
    with open(path, encoding='UTF-8') as f:
        y = yaml.load(f, Loader=yaml.FullLoader)
    if specific is None :
        return y
    else :
        return y[specific]



def read_label_weight(path, cat=None) :
    weight_list = glob(path)
    if cat is None :
        return list2tensor(list(np.load(weight_list[0], allow_pickle=True))), \
               list2tensor(list(np.load(weight_list[1], allow_pickle=True))), \
               list2tensor(list(np.load(weight_list[2], allow_pickle=True)))
    elif cat == 'head1':
        return list2tensor(list(np.load(weight_list[0], allow_pickle=True)))

    elif cat == 'head2':
        return list2tensor(list(np.load(weight_list[1], allow_pickle=True)))

    elif cat == 'head3':
        return list2tensor(list(np.load(weight_list[2], allow_pickle=True)))

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
