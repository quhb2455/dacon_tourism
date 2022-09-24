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

class FCL(nn.Module) :
    def __init__(self, in_c, out_c, dropout_rate, bn):
        super(FCL, self).__init__()
        if bn:
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

    def _FC(self, in_c, out_c, dropout_rate, bn, last=False):
        if last :
            return [nn.Linear(in_c, out_c)]

        elif bn:
            return [nn.Linear(in_c, out_c),
                   nn.BatchNorm1d(out_c),
                   nn.GELU(),
                   nn.Dropout(dropout_rate)]
        else:
            return [nn.Linear(in_c, out_c),
                   nn.GELU(),
                   nn.Dropout(dropout_rate)]

    def _make_layers(self, num_layers, in_c, out_c, dropout_rate, bn):
        layers = []
        reduce_rate = in_c // num_layers
        mid_c = reduce_rate * (num_layers - 1)
        for num in range(1, num_layers) :
            layers += self._FC(in_c, mid_c, dropout_rate, bn)
            in_c = mid_c
            mid_c -= reduce_rate

        layers += self._FC(in_c, out_c, dropout_rate, bn=False, last=True)
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class ClassifierHead1(nn.Module) :
    def __init__(self, num_layers, **kwargs):
        super(ClassifierHead1, self).__init__()
        self.linear = MLP(num_layers=num_layers, **kwargs)

    def forward(self, x):
        return self.linear(x)

class ClassifierHead2(nn.Module) :
    def __init__(self,  num_mlp, num_layers, **kwargs) :
        super(ClassifierHead2, self).__init__()

        cc = kwargs.pop('out_c')
        print(cc)
        print(kwargs)
        self.linear = [MLP(num_layers=num_layers, out_c=cc[i], **kwargs) for i in range(num_mlp)]

    def forward(self, x, prev_head):
        print(prev_head)
        print(prev_head.item())
        print(type(prev_head.item()))
        return self.linear[int(prev_head.item())](x)

if __name__ == '__main__' :
    # m = BackBone(model_name='swin_base_patch4_window7_224_in22k').to('cuda')
    # num_layers=3, in_c=1024, out_c=10, dropout_rate=0.4, bn=False
    m = ClassifierHead2(num_mlp=6, num_layers=3, in_c=1024, out_c=[3,4,5,6,7,8], dropout_rate=0.4, bn=False).to("cuda")
    m.eval()
    o = m(torch.randn(2, 1024).to('cuda'), torch.tensor(2, dtype=torch.int32).to('cuda'))
    print(m)
    print(o.shape)

