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
        output = self.linear(x)
        pred = torch.argmax(output, dim=1)
        return output, pred


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

        return pred[1:]


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
    from utils import CATEGORY_CLS_ENCODER
    CC_ENCODER = CATEGORY_CLS_ENCODER(path='./data/train.csv')

    from utils import read_cfg
    c1, c2, c3 = read_cfg('config.yaml')

    m0 = BackBone(model_name='swin_base_patch4_window7_224_in22k').to('cuda')
    m1 = ClassifierHead1(**c1).to("cuda")
    m2 = ClassifierHead2(**c2).to("cuda")
    m3 = ClassifierHead3(**c3).to("cuda")

    data = torch.rand((16,3,224,224)).to('cuda')
    print(data.shape)

    feature = m0(data)
    output, pred = m1(feature)

    print('car 1 pred : ', pred)
    # batch_num_cat2 = torch.where(((pred == 0) | (pred == 1) | (pred == 3)))
    batch_num_cat3 = torch.where(((pred == 2) | (pred == 4) | (pred == 5)))[0]

    # cat 1 pred 에서 label 1을 출력한 batch 찾아내기
    batch_num_cat2_label03 = torch.where(((pred == 0) | (pred == 3)))[0]
    batch_num_cat2_label1 = torch.where((pred == 1))[0]
    batch_num_cat2, _ = torch.sort(torch.cat([batch_num_cat2_label03, batch_num_cat2_label1], dim=0))
    print("batch_num_cat2 : ", batch_num_cat2)
    print("batch_num_cat2_label1 : ", batch_num_cat2_label1)
    print("batch_num_cat2_label03 : ", batch_num_cat2_label03)
    print()
    cat2_in_label1_batch_num = torch.tensor([0]).to("cuda")
    for i in batch_num_cat2_label1 :
        cat2_in_label1_batch_num = torch.cat([cat2_in_label1_batch_num, torch.where(batch_num_cat2==i)[0]])
    print("cat2_in_label1_batch_num[1:] : ", cat2_in_label1_batch_num[1:])
    cat2_in_label1_batch_num = cat2_in_label1_batch_num[1:]

    print()
    # cat 2에 들어갈 batch들 골라내서 cat2에 입력
    cat2_prev_head = pred[batch_num_cat2]
    cat2_feature = feature[batch_num_cat2]
    print("cat2_prev_head : ",cat2_prev_head)
    print("cat2_feature.shape : ", cat2_feature.shape)
    x2 = m2(cat2_feature, cat2_prev_head)
    print()
    print("x2 : ", x2)
    print()
    # cat 2 pred에서 label 1, label 4 출력한 batch 찾아내서 제거 후 cat 3 입력으로 주기
    # # 전체 batch에서 레포츠 - 복합레포츠 and 레포츠 - 레포츠소개 걸러내기
    batch_num_cat2_label14 = torch.where(((x2[cat2_in_label1_batch_num]==1)|(x2[cat2_in_label1_batch_num]==4)))[0]
    cat2_batch_num_label14 = cat2_in_label1_batch_num[batch_num_cat2_label14]
    batch_num_label14 = batch_num_cat2[cat2_batch_num_label14]
    # cat1_cls1_cat2_cls14 =
    print()
    print("batch_num_cat2_label14 : ", batch_num_cat2_label14)
    print("cat2_batch_num_label14 : ", cat2_batch_num_label14)
    print("batch_num_label14 : ", batch_num_label14)
    # print("cat1_cls1_cat2_cls14 : ", cat1_cls1_cat2_cls14)

    ch = torch.arange(16).to("cuda")
    for i in batch_num_label14 :
        ch = ch[ch != i]
    print("ch : ",  ch)
    print('feature[ch].shape : ', feature[ch].shape)
    print()
    cat3_feature = feature[ch]

    if cat2_batch_num_label14.nelement() != 0 :
        cat2_batch_num_label14_ch = torch.arange(x2.shape[0]).to("cuda")
        print("cat2_batch_num_label14.shape : ", cat2_batch_num_label14.shape)
        print("cat2_batch_num_label14_ch: ", cat2_batch_num_label14_ch)
        for i in cat2_batch_num_label14 :
            cat2_batch_num_label14_ch = cat2_batch_num_label14_ch[cat2_batch_num_label14_ch != i]
        print("cat2_batch_num_label14_ch : ", cat2_batch_num_label14_ch)
        # cat1 -> cat2 에서 복합 레포츠, 레포츠소개 걸러낸 cls값
        no_need_cat3_bn = x2[cat2_batch_num_label14_ch]
        print("no_need_cat3_bn : ", no_need_cat3_bn)


    # cat1에서 label2, label4, label5 속한 값들 cat3의 5, 14, 15번 클래스로 변경
    print("batch_num_cat3 : ", batch_num_cat3)
    print("pred[batch_num_cat3] : ", pred[batch_num_cat3])
    cat1_cls = pred[batch_num_cat3]
    cat1_cat3_cls = CC_ENCODER(prev=None,cur=cat1_cls,category=1)
    print("cat1_cls : ", cat1_cls)
    print('cat3_cls : ', cat1_cat3_cls)
    print()
    # cat1에서 label0, label1, label3 속해서 cat2의 출력으로 나온 값들 cat3 클래스로 변경
    # cat2 통과해서 나온 값들 cat3의 클래스로 넣어주기기
    cat1_cls = pred[batch_num_cat2].detach().cpu().numpy().tolist()
    cat2_cls = x2.detach().cpu().numpy().tolist()
    print("cat1_cls : ", cat1_cls)
    print("cat2_cls : ", cat2_cls)
    cat1_cat2_cat3_cls = CC_ENCODER(prev=cat1_cls, cur=cat2_cls, category=2)
    print('cat3_cls : ', cat1_cat2_cat3_cls)

    # cat1 ->  cat2 -> cat3이 될 값의 원래 배치 번호에 맞게 자리에 넣기
    # batch_num_cat2 = batch_num_cat2.detach().cpu().numpy().tolist()
    # batch_num_cat3 = batch_num_cat3.detach().cpu().numpy().tolist()
    if batch_num_cat2_label14.nelement() != 0:
        batch_num_cat2 = batch_num_cat2[cat2_batch_num_label14_ch]
        cat1_cat2_cat3_cls = torch.tensor(cat1_cat2_cat3_cls, dtype=torch.int32)[cat2_batch_num_label14_ch]


    cat3_input_align = torch.zeros(16, dtype=torch.int32)

    if batch_num_cat2.detach().cpu().numpy().tolist() and batch_num_cat3.detach().cpu().numpy().tolist() :
        batch_num_cat2_cat3 = torch.cat([batch_num_cat2, batch_num_cat3])
        cls = torch.cat([cat1_cat2_cat3_cls, cat1_cat3_cls])

    if batch_num_cat2.detach().cpu().numpy().tolist() :
        if batch_num_cat2_label14.nelement() == 0:
            cat3_input_align[batch_num_cat2] = torch.tensor(cat1_cat2_cat3_cls, dtype=torch.int32)
        else :
            cat3_input_align[batch_num_cat2[cat2_batch_num_label14_ch]] = torch.tensor(cat1_cat2_cat3_cls, dtype=torch.int32)[cat2_batch_num_label14_ch]

    if batch_num_cat3.detach().cpu().numpy().tolist() :
        cat3_input_align[batch_num_cat3] = torch.tensor(cat1_cat3_cls, dtype=torch.int32)
    print("cat3_input_align : ", cat3_input_align)
    print("cat3_input_align.shape : ", cat3_input_align.shape)


     # cat3_prev_head =

