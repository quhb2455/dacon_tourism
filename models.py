import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel

from loss_fn import FocalLoss, AsymmetricLoss

import timm

class FeatureFlatten(nn.Module) :
    def __init__(self, model_type='transformer') :
        super(FeatureFlatten, self).__init__()
        self.model_type = model_type
        self.pool = nn.AdaptiveAvgPool1d(1) if model_type == 'transformer' else nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x = self.pool(x.transpose(2, 1) if self.model_type=='transformer' else x)
        x = self.flatten(x)
        return x


class BackBone(nn.Module):
    def __init__(self, model_name, model_type='transformer', pooling=True):
        super(BackBone, self).__init__()
        self.backbone = timm.create_model(model_name=model_name, num_classes=0, pretrained=True, global_pool='')
        self.feature_flatten = FeatureFlatten(model_type=model_type)
        self.pooling = pooling

    def forward(self, x):
        x = self.backbone.forward_features(x)
        if self.pooling :
            x = self.feature_flatten(x)
        return x


class CNN(nn.Module) :
    def __init__(self,
                 model_name,
                 model_type,
                 head_config,
                 head_name,
                 mode=None):
        super(CNN, self).__init__()

        self.backbone = BackBone(model_name, model_type)
        self.head_name = head_name
        self.mode = mode
        if head_name == 'head1':
            self.cls = ClassifierHead1(**head_config)
        elif head_name == 'head2':
            self.cls = ClassifierHead2(**head_config)
        elif head_name == 'head3':
            self.cls = ClassifierHead3(**head_config)


    def forward(self, x, mask=None):
        feature = self.backbone(x)
        if self.head_name == 'head1' :
            output = self.cls(feature)
        else :
            output = self.cls(feature, mask=mask, mode=self.mode)
        return output


class FCL(nn.Module) :
    def __init__(self, in_c, out_c, dropout_rate, bn, ln, last=False):
        super(FCL, self).__init__()
        if last :
            self.fc = nn.Linear(in_c, out_c)
        elif bn:
            self.fc = nn.Sequential(nn.Linear(in_c, out_c),
                                    nn.BatchNorm1d(out_c),
                                    nn.GELU(),
                                    nn.Dropout(dropout_rate))
        elif ln :
            self.fc = nn.Sequential(nn.Linear(in_c, out_c),
                                    nn.LayerNorm(out_c),
                                    nn.GELU(),
                                    nn.Dropout(dropout_rate))
        else:
            self.fc = nn.Sequential(nn.Linear(in_c, out_c),
                                    nn.GELU(),
                                    nn.Dropout(dropout_rate))
    def forward(self, x):
        return self.fc(x)

class MLP(nn.Module) :
    def __init__(self, num_layers, **kwargs):
        super(MLP, self).__init__()
        self.mlp = self._make_layers(num_layers, **kwargs)

    def _make_layers(self, num_layers, in_c, out_c, dropout_rate, bn, ln):
        layers = []
        reduce_rate = in_c // num_layers
        mid_c = reduce_rate * (num_layers - 1)
        for num in range(1, num_layers) :
            layers += [FCL(in_c, mid_c, dropout_rate, bn, ln)]
            in_c = mid_c
            mid_c -= reduce_rate

        layers += [FCL(in_c, out_c, dropout_rate, bn=False, ln=False, last=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# 대분류 - 1개 -> 6개
class ClassifierHead1(nn.Module) :
    def __init__(self, num_layers, **kwargs):
        super(ClassifierHead1, self).__init__()
        self.linear = MLP(num_layers=num_layers, **kwargs)

    def forward(self, x):
        return self.linear(x)


# 중분류 - 6개 -> 18개
class ClassifierHead2(nn.Module) :
    def __init__(self, num_layers, **kwargs) :
        super(ClassifierHead2, self).__init__()
        self.linear = MLP(num_layers=num_layers, **kwargs)

    def forward(self, x, mask=None, mode='weight'):
        output = self.linear(x)

        if mode == 'mask':
            output.masked_fill_(mask, -10)

        elif mode == 'weight':
            output = output * mask

        return output


# 소분류 - 18개 -> 128개
class ClassifierHead3(nn.Module) :
    def __init__(self, num_layers, **kwargs):
        super(ClassifierHead3, self).__init__()
        self.linear = MLP(num_layers=num_layers, **kwargs)

    def forward(self, x, mask=None, mode='weight'):
        output = self.linear(x)

        if mode == 'mask':
            output.masked_fill_(mask, -10)

        elif mode == 'weight':
            output = output * mask

        return output

class TransformerMODEL(nn.Module):
    def __init__(self, model_name, num_classes=128):
        super(TransformerMODEL, self).__init__()
        self.model = timm.create_model(model_name=model_name, num_classes=num_classes, pretrained=True)

    def forward(self, x):
        x = self.model(x)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self,
                 model_name,
                 model_type,
                 head_config,
                 mode=None):
        super(TransformerClassifier, self).__init__()
        self.backbone = BackBone(model_name, model_type)
        self.mode = mode

        self.cls1 = ClassifierHead1(**head_config['head1'])
        self.cls2 = ClassifierHead2(**head_config['head2'])
        self.cls3 = ClassifierHead3(**head_config['head3'])

    def forward(self, x):
        x = self.backbone(x)

        o1 = self.cls1(x)
        o2 = self.cls2(x, mode=self.mode)
        o3 = self.cls3(x, mode=self.mode)

        return o1, o2, o3

class TextModel(nn.Module):
    def __init__(self, model_name="klue/roberta-base"):
        super(TextModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.model.gradient_checkpointing_enable()

    def forward(self, x, attention_mask):
        x = self.model(input_ids=x, attention_mask=attention_mask)
        return x


class MultiTaskModel(nn.Module) :
    def __init__(self,
                 txt_model_name="klue/roberta-base",
                 head1_cfg=None,
                 head2_cfg=None,
                 head3_cfg=None):

        super(MultiTaskModel, self).__init__()
        self.nlp = TextModel(model_name=txt_model_name)
        self.cls_head1 = ClassifierHead1(**head1_cfg)
        self.cls_head2 = ClassifierHead2(**head2_cfg)
        self.cls_head3 = ClassifierHead3(**head3_cfg)


    def forward(self, txt, att_mask):
        # img_output = self.cnn(img)
        nlp_output = self.nlp(txt, att_mask)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.nlp.model.config.hidden_size, nhead=8).to("cuda")
        # transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to("cuda")

        # concat_output = torch.cat([img_output, nlp_output.last_hidden_state], dim=1)
        # align_output = transformer_encoder(concat_output)
        # align_output = align_output[:, 0]
        align_output = nlp_output.last_hidden_state[:, 0]
        pred1 = self.cls_head1(align_output)
        pred2 = self.cls_head2(align_output, mode=None)
        pred3 = self.cls_head3(align_output, mode=None)
        return pred1, pred2, pred3


if __name__ == '__main__' :
    from utils import read_cfg
    from utils import CATEGORY_MASKING, CLS_WEIGHT, tensor2list
    from datasets import *

    cls_config = read_cfg('./config/cls_config.yaml')
    loss_config = read_cfg('./config/loss_config.yaml')

    csv_path = './data/train.csv'
    img_path = './data/image/train/*'
    batch_size = 16
    device = "cuda"


    img_set, label_set, transform = image_label_dataset(csv_path,
                                                       resize=224,
                                                       training=True)
    train_loader = custom_dataload(img_set,
                                    label_set,
                                    csv_path,
                                    transform, batch_size=batch_size, shuffle=True)

    model = MultiTaskModel(img_model_name='vit_base_patch32_224',
                           txt_model_name='klue/roberta-base',
                           head1_cfg=cls_config['head1'],
                           head2_cfg=cls_config['head2'],
                           head3_cfg=cls_config['head3']).to("cuda")
    model.train()
    # cls_weight_generator = CLS_WEIGHT(csv_path=csv_path)#CATEGORY_MASKING(path=csv_path)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

    for batch, (img, txt, att_mask, label1, label2, label3) in enumerate(train_loader, start=1):
        img = img.to(device)
        txt = txt.to(device)
        att_mask = att_mask.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)
        label3 = label3.to(device)

        o1, o2, o3 = model(img, txt, att_mask)
        print(o1.shape)
        print(o2.shape)
        print(o3.shape)


        break

