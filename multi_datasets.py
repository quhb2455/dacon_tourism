import os

import cv2
from glob import glob
import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoTokenizer

import torch
from torch.utils.data import Dataset, DataLoader

from utils import LABEL_ENCODER

class CustomDataset(Dataset):
    def __init__(self, img_list, label_set=None, path=None, transforms=None):
        self.img_list = img_list
        self.label_set = label_set
        self.max_len=256
        self.text_list = pd.read_csv(path)['overview'].values
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
        self.transforms = transforms

        # if label_set is not None :
        #     label_enc = LABEL_ENCODER(path)
        #     self.cat1_enc = label_enc.cat1_label_encoder()
        #     self.cat2_enc = label_enc.cat2_label_encoder()
        #     self.cat3_enc = label_enc.cat3_label_encoder()
        #     self.cat2_ig_enc = label_enc.cat2_label_index_encoder()
        #     self.cat3_ig_enc = label_enc.cat3_label_index_encoder()
        #
        #     self.cat1_label_list, self.cat2_label_list, self.cat3_label_list = self.label_encoder(label_set)


    def __len__(self):
        # if self.label_set is not None :
        #     assert len(self.img_list) == len(self.cat1_label_list), 'must be same length between img_lst and label_list'

        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]

        img = cv2.imread(os.path.join('./data/',img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        text = self.text_list[idx]
        text_data = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        if self.transforms:
            img = self.transforms(image=img)['image']

        # training
        if self.label_set is not None:
            cat1_label = int(self.label_set['cat1'].iloc[idx])
            cat2_label = int(self.label_set['cat2'].iloc[idx])
            cat3_label = int(self.label_set['cat3'].iloc[idx])

            return img, \
                   text_data['input_ids'].flatten(), text_data['attention_mask'].flatten(), \
                   torch.tensor(cat1_label), torch.tensor(cat2_label), torch.tensor(cat3_label)

        # test
        else:
            return img, text_data['input_ids'].flatten(), text_data['attention_mask'].flatten()

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

def label_mask(label, ig_enc, size):
    # masking에서 True는 적용하는 값 False는 적용안하는 값
    label2_mask = [True] * size
    for v1 in ig_enc[label].values() :
        label2_mask[v1] = False
        for v2 in cat3_ig_enc[v1].values() :
            label3_mask[v2] = False
    return label2_mask, label3_mask


def transform_parser(resize=224) :
    return A.Compose([
        # A.Resize(250, 375),
        # A.OneOf([
        #     A.RandomCrop(resize, resize),
        #     A.CenterCrop(resize, resize)
        # ], p=1),
        # A.OneOf([
        #     A.ToGray(),
        #     A.ChannelShuffle(),
        # ], p=0.5),
        A.Resize(resize+30, resize+30),
        A.RandomCrop(resize, resize),
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.RandomRotate90(p=1),
            A.VerticalFlip(p=1)
        ], p=1),
        A.Normalize(),
        ToTensorV2()
    ])


def img_parser(data_path, training=True):
    #
    # if training:
    #     return path[:div], path[div:]
    # else:
    #     return path
    if training:
        return list(data_path)
    else:
        path = sorted(list(map(lambda x: x.replace(x.split('.')[0], data_path.split('.')[0]), glob(data_path))),
                      key=lambda x: int(x.split('\\')[-1].split('.')[0].split('_')[-1]))
        return list(path)


def label_parser(df) :
    df = pd.concat([df['cat1'], df['cat2'], df['cat3']], axis=1)
    return df


def image_label_dataset(df_path, resize=224, training=True):
    all_df = pd.read_csv(df_path)
    transform = transform_parser(resize=resize)

    if training:
        img_list = img_parser(all_df['img_path'], training=training)
        label_df = label_parser(all_df)
        return img_list, label_df, transform

    else:
        img = img_parser(all_df['img_path'], training=training)
        return img, all_df, transform


def custom_dataload(img_set, label_set, csv_path, transform, batch_size, shuffle) :
    ds = CustomDataset(img_set, label_set , csv_path, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl


def train_and_valid_dataload(img_set, label_set, csv_path, transform, batch_size=16) :
    train_loader = custom_dataload(img_set[0], label_set[0], csv_path, transform, batch_size, True)
    if len(img_set[1]) == 0 :
        return train_loader, None
    else :
        val_loader = custom_dataload(img_set[1], label_set[1], csv_path, transform, batch_size, True)
        return train_loader, val_loader


if __name__ == "__main__" :
    csv_path = './data/train.csv'
    img_path = './data/image/train/*'
    batch_size = 16

    img_set, label_set, transform = image_label_dataset(csv_path,
                                                       resize=224,
                                                       training=True)

    train_loader, val_loader = train_and_valid_dataload(img_set, label_set, csv_path, transform, batch_size=batch_size)

    print(val_loader)
    cnt = 0
    for img, label1, label2, label3 in train_loader :

        print(label1, label2, label3)
        if cnt == 3 :
            break
        cnt += 1

    cnt = 0
    print("???")
    for img, label1, label2, label3 in val_loader :

        print(label1, label2, label3)
        if cnt == 3 :
            break
        cnt += 1

    cnt = 0
    # for img, label1, label2, label3 in val_loader :
    #     print(label1, label2, label3)
    #     break
