import torch
from torch import nn
import torch.nn.functional as F
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
    def __init__(self, num_layers, **kwargs):
        super(ClassifierHead1, self).__init__()
        self.linear = MLP(num_layers=num_layers, **kwargs)

    def forward(self, x):
        return self.linear(x)


# 중분류 - 6개 -> 18개
class ClassifierHead2(nn.Module) :
    def __init__(self,  num_mlp, num_layers, out_c, **kwargs) :
        super(ClassifierHead2, self).__init__()
        self.linear = nn.ModuleList([MLP(num_layers=num_layers, out_c=out_c[i], **kwargs) for i in range(num_mlp)])

    def forward(self, x, prev_head):
        pred = torch.zeros(1, dtype=torch.int32).to('cuda')
        for i in range(0, prev_head.shape[0]) :
            prev_head[i] = 2 if prev_head[i] == 3 else prev_head[i]
            output = self.linear[prev_head[i]](x[i])
            arg = torch.argmax(output, dim=0)
            pred = torch.cat([pred, arg.view(-1)], dim=0)
            print("arg : ", arg)

        print("pred : ", pred)
        print("pred[1:] : ", pred[1:])
        print('== cat2 done ==')
        return self.linear[prev_head](x)


# 소분류 - 18개 -> 128개
class ClassifierHead3(nn.Module) :
    def __init__(self, num_mlp, num_layers, out_c, **kwargs):
        super(ClassifierHead3, self).__init__()
        self.liner = nn.ModuleList([MLP(num_layers=num_layers, out_c=out_c[i], **kwargs) for i in range(num_mlp)])

    def forward(self, x, prev_head):
        return self.liner[prev_head](x)


if __name__ == '__main__' :
    m = BackBone(model_name='swin_base_patch4_window7_224_in22k').to('cuda')
    # m1 = ClassifierHead1(num_layers=3, in_c=1024, out_c=6, dropout_rate=0.4, bn=False).to("cuda")
    # m2 = ClassifierHead2(num_mlp=3, num_layers=3, in_c=1024, out_c=[2, 5, 8], dropout_rate=0.4, bn=False).to("cuda")
    # m3 = ClassifierHead3(num_mlp=16, num_layers=3, in_c=1024, out_c=[18, 2, 19, 8, 2, 8, 2, 10, 14, 8, 8, 3, 6, 3, 9, 7], dropout_rate=0.4, bn=False).to("cuda")

    from utils import read_cfg
    c1, c2, c3 = read_cfg('config.yaml')

    m0 = BackBone(model_name='swin_base_patch4_window7_224_in22k').to('cuda')
    m1 = ClassifierHead1(**c1).to("cuda")
    m2 = ClassifierHead2(**c2).to("cuda")
    m3 = ClassifierHead3(**c3).to("cuda")

    data =torch.rand((16,3,224,224)).to('cuda')
    print(data.shape)

    feature = m0(data)
    print(feature.shape)
    x1 = m1(feature)

    head = torch.argmax(x1, dim=1)
    print('head : ', head)
    batch_num_cat2 = torch.where(((head == 0) | (head == 1) | (head == 3)))
    batch_num_cat3 = torch.where(((head == 2) | (head == 4) | (head == 5)))

    cat2_prev_head = head[batch_num_cat2]
    cat2_feature = feature[batch_num_cat2]
    print("cat2_prev_head : ",cat2_prev_head)
    print("cat2_feature.shape : ", cat2_feature.shape)
    x2 = m2(cat2_feature, cat2_prev_head)

    cat3_feature = feature[batch_num_cat3]


    # print(torch.where(((head==0)|(head==1)|(head==3))))
    # print(torch.argmax(x1, dim=1))
    # print(x1.shape)


    # print(x2)
    # print(x2.shape)
