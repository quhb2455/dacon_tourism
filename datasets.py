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
            self.cat1_label_list, self.cat2_label_list, self.cat3_label_list = self.label_encoder(path, label_set)
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


            return img, torch.tensor(cat1_label), torch.tensor(cat2_label),  torch.tensor(cat3_label)

        # test
        else:
            return img

    def label_encoder(self, path, label_set):
        label_enc = LABEL_ENCODER(path)
        cat1_enc = label_enc.cat1_label_encoder()
        cat2_enc = label_enc.cat2_label_encoder()
        cat3_enc = label_enc.cat3_label_encoder()

        label_dec = CATEGORY_CLS_ENCODER(path)
        cat2_dec = label_dec.cat2_cls_encoder()

        cat1_label_list = list(map(lambda x: cat1_enc[x], label_set['cat1']))
        cat2_label_list = list(map(lambda x, y: cat2_enc[y][x], label_set['cat2'], cat1_label_list))
        cat3_label_list = list(map(lambda x, y, z: cat3_enc[cat2_dec[z][y]][x], label_set['cat3'], cat2_label_list, cat1_label_list))

        return cat1_label_list, cat2_label_list, cat3_label_list

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
        break

    cnt = 0
    for img, label1, label2, label3 in val_loader :
        print(label1, label2, label3)
        break
