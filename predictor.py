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
from models import BackBone, ClassifierHead1, ClassifierHead2, ClassifierHead3
from loss_fn import FocalLoss

class Predictor() :
    def __init__ (self, backbone, head1, head2, head3, device, args) :
        self.backbone, self.head1, self.head2, self.head3 = get_models(backbone,
                                                                       head1,
                                                                       head2,
                                                                       head3,
                                                                       args.CHECKPOINT)
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
        self.backbone.eval()
        self.head1.eval()
        self.head2.eval()
        self.head3.eval()

        model_preds = []
        with torch.no_grad() :
            for img in tqdm(self.test_loader) :
                img = img.to(self.device)

                feature_map = self.backbone(img)
                pred1 = self.head1(feature_map)  # [batch, 6]


                mask2 = self.cls_weight_generator(pred1, num=18, category=1)
                pred2 = self.head2(feature_map, mask2, mode='weight')  # [batch, 18]

                mask3 = self.cls_weight_generator(pred2, num=128, category=2)
                pred3 = self.head3(feature_map, mask3, mode='weight')  # # [batch, 128]

                model_preds += tensor2list(pred3.argmax(1))

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
    parser.add_argument("--MODEL_NAME", type=str, default='efficientnetv2_rw_m')
    parser.add_argument("--MODEL_TYPE", type=str, default='cnn')
    parser.add_argument("--RESIZE", type=int, default=224)

    parser.add_argument("--ENSEMBLE", type=str, default=None)
    parser.add_argument("--IMG_PATH", type=str, default="./data/image/test/*")
    parser.add_argument("--CSV_PATH", type=str, default="./data/train.csv")
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

    backbone = BackBone(model_name=args.MODEL_NAME, model_type=args.MODEL_TYPE).to('cuda')
    head1 = ClassifierHead1(**cls_config['head1']).to("cuda")
    head2 = ClassifierHead2(**cls_config['head2']).to("cuda")
    head3 = ClassifierHead3(**cls_config['head3']).to("cuda")

    predictor = Predictor(backbone, head1, head2, head3, device, args)

    preds = predictor.run()

    preds = [label_decode[p] for p in preds]
    save_to_csv(predictor.df, preds, args.OUTPUT)