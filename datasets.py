import cv2
from glob import glob
import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader

from utils import LABEL_ENCODER, CATEGORY_CLS_ENCODER

class CustomDataset(Dataset):
    def __init__(self, img_list, label_set=None, path=None, transforms=None):
        self.img_list = img_list

        if label_set is not None :
            label_enc = LABEL_ENCODER(path)
            self.cat1_enc = label_enc.cat1_label_encoder()
            self.cat2_enc = label_enc.cat2_label_encoder()
            self.cat3_enc = label_enc.cat3_label_encoder()
            self.cat2_ig_enc = label_enc.cat2_label_index_encoder()
            self.cat3_ig_enc = label_enc.cat3_label_index_encoder()

            self.cat1_label_list, self.cat2_label_list, self.cat3_label_list = self.label_encoder(label_set)
        self.transforms = transforms

    def __len__(self):
        assert len(self.img_list) == len(self.cat1_label_list), 'must be same length between img_lst and label_list'
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(image=img)['image']

        # training
        if self.cat1_label_list is not None:
            cat1_label = self.cat1_label_list[idx]
            cat2_label = self.cat2_label_list[idx]
            cat3_label = self.cat3_label_list[idx]

            cat2_mask, cat3_mask = self.label_ignore(cat1_label)

            return img, torch.tensor(cat1_label), \
                   (torch.tensor(cat2_label), torch.tensor(cat2_mask, dtype=torch.bool)), \
                   (torch.tensor(cat3_label), torch.tensor(cat3_mask, dtype=torch.bool))
        # test
        else:
            return img

    def label_encoder(self, label_set):
        cat1_label_list = list(map(lambda x: self.cat1_enc[x], label_set['cat1']))
        cat2_label_list = list(map(lambda x: self.cat2_enc[x], label_set['cat2']))
        cat3_label_list = list(map(lambda x: self.cat3_enc[x], label_set['cat3']))
        return cat1_label_list, cat2_label_list, cat3_label_list

    def label_ignore(self, label1):
        # masking에서 True는 적용하는 값 False는 적용안하는 값
        label2_mask = [True] * 18
        label3_mask = [True] * 128
        for v1 in self.cat2_ig_enc[label1].values() :
            label2_mask[v1] = False
            for v2 in self.cat3_ig_enc[v1].values() :
                label3_mask[v2] = False

        return label2_mask, label3_mask

def label_mask(label1, cat2_ig_enc, cat3_ig_enc):
    # masking에서 True는 적용하는 값 False는 적용안하는 값
    label2_mask = [True] * 18
    label3_mask = [True] * 128
    for v1 in cat2_ig_enc[label1].values() :
        label2_mask[v1] = False
        for v2 in cat3_ig_enc[v1].values() :
            label3_mask[v2] = False

    return label2_mask, label3_mask

def transform_parser(resize=224) :
    return A.Compose([
        A.Rotate(limit=(45), p=1),
        # A.RandomGridShuffle(p=grid_shuffle_p, grid=(2,2)),
        A.Resize(resize, resize),
        A.Normalize(),
        ToTensorV2()
    ])


def img_parser(data_path, div, training=True):
    path = sorted(list(map(lambda x : x.replace(x.split('.')[0], data_path.split('.')[0]), glob(data_path))),
                  key=lambda x: int(x.split('\\')[-1].split('.')[0].split('_')[-1]))

    if training:
        return path[:div], path[div:]
    else:
        return path


def label_parser(df, div) :
    df = pd.concat([df['cat1'], df['cat2'], df['cat3']], axis=1)
    return df[:div], df[div:]


def image_label_dataset(df_path, img_path, div=0.8, resize=224, training=True):
    all_df = pd.read_csv(df_path)
    transform = transform_parser(resize=resize)

    if training:
        train_img, valid_img = img_parser(img_path, int(len(all_df) * div), training=training)
        train_label_df, valid_label_df = label_parser(all_df, int(len(all_df) * div))
        return (train_img, valid_img), (train_label_df, valid_label_df), transform

    else:
        img = img_parser(img_path, div=None, training=training)
        return img, all_df, transform


def custom_dataload(img_set, label_set, csv_path, transform, batch_size, shuffle) :
    ds = CustomDataset(img_set, label_set , csv_path, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl


def train_and_valid_dataload(img_set, label_set, csv_path, transform, batch_size=16) :
    train_loader = custom_dataload(img_set[0], label_set[0], csv_path, transform, batch_size, True)
    val_loader = custom_dataload(img_set[1], label_set[1], csv_path, transform, batch_size, False)
    return train_loader, val_loader


if __name__ == "__main__" :
    csv_path = './data/train.csv'
    img_path = './data/image/train/*'
    batch_size = 16

    img_set, label_set, transform = image_label_dataset(csv_path,
                                                       img_path,
                                                       div=0.8,
                                                       resize=224,
                                                       training=True)

    train_loader, val_loader = train_and_valid_dataload(img_set, label_set, csv_path, transform, batch_size=batch_size)


    cnt = 0
    for img, label1, label2, label3 in train_loader :

        print(label1, label2, label3)
        if cnt == 3 :
            break
        cnt += 1

    cnt = 0
    # for img, label1, label2, label3 in val_loader :
    #     print(label1, label2, label3)
    #     break
