import torch
from torch import nn
import torch.nn.functional as F

from loss_fn import FocalLoss, AsymmetricLoss
from utils import CATEGORY_CLS_ENCODER, tensor2list
from datasets import *
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
    def __init__(self, model_name, model_type='transformer'):
        super(BackBone, self).__init__()
        self.backbone = timm.create_model(model_name=model_name, num_classes=0, pretrained=True, global_pool='')
        self.feature_flatten = FeatureFlatten(model_type=model_type)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.feature_flatten(x)
        return x


class CNN(nn.Module) :
    def __init__(self,
                 model_name,
                 model_type,
                 config_path):
        super(CNN, self).__init__()
        c1, c2, c3 = read_cfg(config_path)

        self.backbone = BackBone(model_name, model_type)
        self.cls1 = ClassifierHead1(**c1)
        self.cls2 = ClassifierHead2(**c2)
        self.cls3 = ClassifierHead3(**c3)

    def forward(self, x):
        feature = self.backbone(x)

        cat1 = self.cls1(feature)
        cat2 = self.cls2(feature, cat1.argmax(1))
        cat3 = self.cls3(feature, cat2.argmax(1))

        return cat1, cat2, cat3

class FCL(nn.Module) :
    def __init__(self, in_c, out_c, dropout_rate, bn, last=False):
        super(FCL, self).__init__()
        if last :
            self.fc = nn.Linear(in_c, out_c)
        elif bn:
            self.fc = nn.Sequential(nn.Linear(in_c, out_c),
                                    nn.BatchNorm1d(out_c),
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

    def _make_layers(self, num_layers, in_c, out_c, dropout_rate, bn):
        layers = []
        reduce_rate = in_c // num_layers
        mid_c = reduce_rate * (num_layers - 1)
        for num in range(1, num_layers) :
            layers += [FCL(in_c, mid_c, dropout_rate, bn)]
            in_c = mid_c
            mid_c -= reduce_rate

        layers += [FCL(in_c, out_c, dropout_rate, bn=False, last=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# 대분류 - 1개 -> 6개
class ClassifierHead1(nn.Module) :
    def __init__(self, num_layers, loss_cfg=None, **kwargs):
        super(ClassifierHead1, self).__init__()
        self.linear = MLP(num_layers=num_layers, **kwargs)
        self.focal = FocalLoss(**loss_cfg['focal'])

    def forward(self, x, label=None):
        output = self.linear(x)
        pred = torch.argmax(output, dim=1)
        loss = 0
        if label is not None :
            loss = self.focal(output, label)
            # loss = nn.CrossEntropyLoss()(output, label)
        return pred, loss


# 중분류 - 6개 -> 18개
class ClassifierHead2(nn.Module) :
    def __init__(self,  num_mlp, num_layers, out_c, loss_cfg=None, **kwargs) :
        super(ClassifierHead2, self).__init__()
        self.linear = nn.ModuleList([MLP(num_layers=num_layers, out_c=out_c[i], **kwargs) for i in range(num_mlp)])
        self.asl = AsymmetricLoss(**loss_cfg['asymmetric'])
        self.focal = FocalLoss(**loss_cfg['focal'])

    def forward(self, x, prev_head, label=None):
        pred = torch.zeros(1, dtype=torch.int32).to('cuda')
        loss = 0
        print("cls2 prev head : ",prev_head)
        print("cls2 linear : ", self.linear)
        print("cls2 label : ", label)

        for i, ph in enumerate(prev_head) :
            print("cls2 ph : ", ph)
            output = self.linear[ph](x[i].unsqueeze(0))
            arg = torch.argmax(output, dim=0)
            pred = torch.cat([pred, arg.view(-1)], dim=0)
            if label is not None :
                loss += self.loss_fn(output, label[i])
        return pred[1:], loss

    def loss_fn(self, output, label):
        print("cls2 output : ", output)
        print("cls2 label : ", label)
        if output[0].shape[0] == 1 :
            loss = self.asl(output, label)#torch.tensor([label], dtype=torch.float32).to("cuda"))
            print("2 aysm loss : ", loss)
            # loss = nn.BCEWithLogitsLoss()(output, torch.tensor([label], dtype=torch.float32).to("cuda"))
        else :
            loss = self.focal(output[0], label)
            print("2 focal loss : ", loss)
            # loss = nn.CrossEntropyLoss()(output, label)
        return loss

# 소분류 - 18개 -> 128개
class ClassifierHead3(nn.Module) :
    def __init__(self, num_mlp, num_layers, out_c, loss_cfg=None, **kwargs):
        super(ClassifierHead3, self).__init__()
        self.linear = nn.ModuleList([MLP(num_layers=num_layers, out_c=out_c[i], **kwargs) for i in range(num_mlp)])
        # self.loss_cfg = loss_cfg
        self.asl = AsymmetricLoss(**loss_cfg['asymmetric'])
        self.focal = FocalLoss(**loss_cfg['focal'])

    def forward(self, x, prev_head, label=None):
        pred = torch.zeros(1, dtype=torch.int32).to('cuda')
        loss = 0
        for i, ph in enumerate(prev_head) :
            output = self.linear[ph](x[i].unsqueeze(0))
            arg = torch.argmax(output, dim=0)
            pred = torch.cat([pred, arg.view(-1)], dim=0)
            if label is not None :
                loss += self.loss_fn(output, label[i])
        return pred[1:], loss

    def loss_fn(self, output, label):
        print("cls3 output : ", output)
        print("cls3 label : ", label)
        if output[0].shape[0] == 1 :
            loss = self.asl(output, label)#torch.tensor([label], dtype=torch.float32).to("cuda"))
            # loss = nn.BCEWithLogitsLoss()(output, torch.tensor([label], dtype=torch.float32).to("cuda"))
            print("3 aysm loss : ", loss)
        else :
            loss = self.focal(output[0], label)
            print("3 focal loss : ", loss)
            # loss = nn.CrossEntropyLoss()(output, label)
        return loss

class MultiTaskModel(nn.Module) :
    def __init__(self,
                 model_name='swin_base_patch4_window7_224_in22k',
                 head1_cfg=None,
                 head2_cfg=None,
                 head3_cfg=None,
                 loss_cfg=None,
                 path='./data/train.csv'):

        super(MultiTaskModel, self).__init__()
        self.backbone = BackBone(model_name=model_name)
        self.cls_head1 = ClassifierHead1(loss_cfg=loss_cfg, **head1_cfg)
        self.cls_head2 = ClassifierHead2(loss_cfg=loss_cfg, **head2_cfg)
        self.cls_head3 = ClassifierHead3(loss_cfg=loss_cfg, **head3_cfg)

        self.cls_enc = CATEGORY_CLS_ENCODER(path)
        # self.cls1_enc = self.cls_enc.cat1_cls_encoder()
        # self.cls2_enc = self.cls_enc.cat2_cls_encoder()
        # self.cls3_enc = self.cls_enc.cat3_cls_encoder()

    def forward(self, x, label1=None, label2=None, label3=None):

        feature_map = self.backbone(x)
        pred1, loss1 = self.cls_head1(feature_map, label=label1)
        print("pred1 : ", pred1)
        print("pred1 to pred2 ==> ", self.cls_enc(prev=None, cur=pred1, category=1))

        pred2, loss2 = self.cls_head2(feature_map, pred1, label=label2)
        print("pred2 : ", pred2)
        print("pred2 to pred3 ==> ", self.cls_enc(prev=pred1, cur=pred2, category=2))

        pred3, loss3 = self.cls_head3(feature_map,
                                      self.cls_enc(prev=pred1, cur=pred2, category=2),
                                      label=label3)
        return pred1, pred2, pred3, loss1, loss2, loss3


if __name__ == '__main__' :
    from utils import read_cfg
    cls_config = read_cfg('./config/cls_config.yaml')
    loss_config = read_cfg('./config/loss_config.yaml')

    csv_path = './data/train.csv'
    img_path = './data/image/train/*'
    batch_size = 16

    # m0 = BackBone(model_name='swin_base_patch4_window7_224_in22k').to('cuda')
    # m1 = ClassifierHead1(**c1).to("cuda")
    # m2 = ClassifierHead2(**c2).to("cuda")
    # m3 = ClassifierHead3(**c3).to("cuda")
    img_set, label_set, transform = image_label_dataset(csv_path,
                                                       img_path,
                                                       div=0.8,
                                                       resize=224,
                                                       training=True)
    train_loader, val_loader = train_and_valid_dataload(img_set,
                                                        label_set,
                                                        csv_path,
                                                        transform, batch_size=batch_size)

    model = MultiTaskModel(model_name='swin_base_patch4_window7_224_in22k',
                           head1_cfg=cls_config['head1'],
                           head2_cfg=cls_config['head2'],
                           head3_cfg=cls_config['head3'],
                           loss_cfg=loss_config).to("cuda")
    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    device = "cuda"
    for batch, (img, label1, label2, label3) in enumerate(train_loader, start=1):
        img = img.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)
        label3 = label3.to(device)

        optimizer.zero_grad()
        data = torch.rand((16,3,224,224)).to('cuda')
        print("img.shape : ", img.shape)
        p1, p2, p3, loss1, loss2, loss3 = model(img, label1, label2, label3)
        print(f"loss 1 : {loss1}, loss 2 : {loss2}, loss 3 : {loss3}")
        loss = loss1 + loss2 + loss3
        print(f"loss : {loss}")
        loss.backward()
        optimizer.step()


