import pandas as pd
import torch
import os
import argparse
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

from datasets import *
from utils import *
from models import BackBone, ClassifierHead1, ClassifierHead2, ClassifierHead3, CNN
from loss_fn import FocalLoss

class Predictor() :
    def __init__ (self, head1_cnn, head2_cnn, head3_cnn, device, args) :
        self.head1_cnn = single_model_weight_load(head1_cnn,
                                                 None,
                                                 './ckpt/each_head/1/lossWeight-auged-effiv2s/29E-val0.9343-efficientnetv2_rw_s.pth',
                                                 training=False)
        self.head2_cnn = single_model_weight_load(head2_cnn,
                                                 None,
                                                 './ckpt/each_head/2/head1mask-lossWeight-auged-effiv2s/26E-val0.9051-efficientnetv2_rw_s.pth',
                                                 training=False)
        self.head3_cnn = single_model_weight_load(head3_cnn,
                                                 None,
                                                 './ckpt/each_head/3/head1-2mask-lossWeight-auged-effiv2s/29E-val0.8145-efficientnetv2_rw_s.pth',
                                                 training=False)

        self.test_loader = self.get_dataloader(args.CSV_PATH, args.IMG_PATH, args.BATCH_SIZE, args.RESIZE)
        self.df = pd.read_csv(args.SUB_CSV_PATH)

        self.cls_weight_generator = CLS_WEIGHT(csv_path=args.CSV_PATH)

        self.ensemble = args.ENSEMBLE
        self.save_path = args.OUTPUT
        self.batch_size = args.BATCH_SIZE
        self.device = device

    def run(self):
        if self.ensemble == 'soft':
            return self.ensemble_predict_softVoting()

        elif self.ensemble == 'hard':
            return self.ensemble_predict_hardVoting()

        elif self.ensemble is None:
            return self.predict()


    def predict(self):
        self.head1_cnn.eval()
        self.head2_cnn.eval()
        self.head3_cnn.eval()

        model_preds = []
        with torch.no_grad() :
            for img in tqdm(self.test_loader) :
                img = img.to(self.device)

                mask1_pred = self.head1_cnn(img)
                mask1 = self.cls_weight_generator(mask1_pred, num=18, category=1)
                mask2_pred = self.head2_cnn(img, mask=mask1)
                mask2 = self.cls_weight_generator(mask2_pred, num=128, category=2)
                pred = self.head3_cnn(img, mask=mask2)

                model_preds += tensor2list(pred.argmax(1))

        return model_preds

    def ensemble_predict_softVoting(self):
        model_preds = []
        with torch.no_grad():
            for img in tqdm(self.test_loader):
                img = img.to(self.device)
                batch_preds_score = []
                for m in self.model:
                    m.eval()
                    pred = m(img)
                    batch_preds_score.append(pred.detach().cpu().numpy())

                batch_preds_score = np.mean(np.array(batch_preds_score), axis=0)
                best_score_ind = np.argmax(batch_preds_score, axis=1)
                model_preds += best_score_ind.tolist()
        return model_preds

    def ensemble_predict_hardVoting(self):
        model_preds = []
        with torch.no_grad():
            for img in tqdm(self.test_loader):
                img = img.to(self.device)

                batch_len = [i for i in range(self.batch_size)]
                batch_preds_score = []
                batch_preds_label = []
                for m in self.model:
                    m.eval()
                    pred = m(img)

                    pred = pred.max(1)
                    batch_preds_score.append(pred[0].detach().cpu().numpy())
                    batch_preds_label.append(pred[1].detach().cpu().numpy())

                best_score_ind = np.argmax(batch_preds_score, axis=0)
                batch_preds_label = np.array(batch_preds_label)

                model_preds += batch_preds_label[best_score_ind[batch_len], batch_len].tolist()
        return model_preds

    def get_dataloader(self, csv_path, img_path, batch_size, resize=224):
        img_set, df, transform = image_label_dataset(csv_path, img_path, resize=resize, training=False)
        return custom_dataload(img_set, None, csv_path, transform, batch_size, shuffle=False)



if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--BATCH_SIZE", type=int, default=32)
    parser.add_argument("--MODEL_NAME", type=str, default='efficientnetv2_rw_s')
    parser.add_argument("--MODEL_TYPE", type=str, default='cnn')
    parser.add_argument("--RESIZE", type=int, default=224)

    parser.add_argument("--ENSEMBLE", type=str, default=None)
    parser.add_argument("--IMG_PATH", type=str, default="./data/image/test/*")
    parser.add_argument("--CSV_PATH", type=str, default="./data/aug_train.csv")
    parser.add_argument("--SUB_CSV_PATH", type=str, default="./data/sample_submission.csv")
    parser.add_argument("--CLS_CONFIG", type=str, default='./config/cls_config.yaml')
    parser.add_argument("--LOSS_CONFIG", type=str, default='./config/loss_config.yaml')

    parser.add_argument("--OUTPUT", type=str, default='./submission/59E-val0.7574-lr0.01_fullData_maskingWeight_efficientnetv2_m.csv')
    parser.add_argument("--CHECKPOINT",  nargs="+", type=str,
                        default=['./ckpt/newlr0.01_fullData_maskingWeight_efficientnetv2_m/59E-val0.7574587666372415-efficientnetv2_rw_m.pth'])


    args = parser.parse_args()
    cls_config = read_cfg(args.CLS_CONFIG)
    loss_config = read_cfg(args.LOSS_CONFIG)
    label_decode = {v: k for k, v in LABEL_ENCODER(args.CSV_PATH).cat3_label_encoder().items()}

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # head 1 setup
    head1_cnn = CNN(model_name=args.MODEL_NAME,
                    model_type=args.MODEL_TYPE,
                    head_config=cls_config['head1'],
                    head_name='head1').to("cuda")

    # head 2 setup
    head2_cnn = CNN(model_name=args.MODEL_NAME,
                    model_type=args.MODEL_TYPE,
                    head_config=cls_config['head2'],
                    head_name='head2',
                    mode='weight').to("cuda")

    # head 3 setup
    head3_cnn = CNN(model_name=args.MODEL_NAME,
                    model_type=args.MODEL_TYPE,
                    head_config=cls_config['head3'],
                    head_name='head3',
                    mode='weight').to("cuda")

    predictor = Predictor(head1_cnn, head2_cnn, head3_cnn, device, args)

    preds = predictor.run()

    preds = [label_decode[p] for p in preds]
    save_to_csv(predictor.df, preds, args.OUTPUT)