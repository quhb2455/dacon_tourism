import torch
import os
import argparse
from tqdm import tqdm

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from datasets import *
from utils import *
from models import *
from loss_fn import FocalLoss

class Trainer() :
    def __init__ (self, model, criterion, optimizer, lr_scheduler, device, args) :
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.log_writter = SummaryWriter(args.LOG)
        self.save_path = args.OUTPUT
        self.args = args
        self.best_score = 0
        self.device = device
        self.APPLY_MIXUP = False

    def run(self):
        if self.args.REUSE :
            self.model, self.optimizer, self.args.START_EPOCH = weight_load(self.model, self.optimizer, self.args.CHECKPOINT)

        for epoch in range(self.args.START_EPOCH + 1, self.args.EPOCHS + 1) :

            # training
            self.training(epoch)

            # validation
            self.validation(epoch)


    def training(self, epoch):
        self.model.train()

        tqdm_train = tqdm(self.train_loader)
        train_acc, train_loss = [], []
        # for batch, (img, label1, (label2, mask2), (label3, mask3)) in enumerate(tqdm_train, start=1):
        # for batch, (img, txt, att_mask, label1, label2, label3) in enumerate(tqdm_train, start=1):
        for batch, (img, label1, label2, label3) in enumerate(tqdm_train, start=1):

            self.optimizer.zero_grad()

            img = img.to(device)
            # txt = txt.to(device)
            # att_mask = att_mask.to(device)
            label1 = label1.to(device)
            label2 = label2.to(device)
            label3 = label3.to(device)

            if self.args.AUG:
                p = np.random.rand(1)

                # mixup
                if p < 0.5 :
                    img, lam, label_a1, label_b1, label_a2, label_b2, label_a3, label_b3 = mixup(img,
                                                                                               label1,
                                                                                               label2,
                                                                                               label3)
                    pred1, pred2, pred3 = self.model(img)
                    loss1 = lam * self.criterion(pred1, label_a1) + (1 - lam) * self.criterion(pred1, label_b1)
                    loss2 = lam * self.criterion(pred2, label_a2) + (1 - lam) * self.criterion(pred2, label_b2)
                    loss3 = lam * self.criterion(pred3, label_a3) + (1 - lam) * self.criterion(pred3, label_b3)

                # cutmix
                else :
                    img, lam, label_a1, label_b1, label_a2, label_b2, label_a3, label_b3 = cutmix(img,
                                                                                                label1,
                                                                                                label2,
                                                                                                label3)
                    pred1, pred2, pred3 = self.model(img)
                    loss1 = lam * self.criterion(pred1, label_a1) + (1 - lam) * self.criterion(pred1, label_b1)
                    loss2 = lam * self.criterion(pred2, label_a2) + (1 - lam) * self.criterion(pred2, label_b2)
                    loss3 = lam * self.criterion(pred3, label_a3) + (1 - lam) * self.criterion(pred3, label_b3)
            else:
                pred1, pred2, pred3 = self.model(img)
                loss1 = self.criterion(pred1, label1)
                loss2 = self.criterion(pred2, label2)
                loss3 = self.criterion(pred3, label3)

            loss = (loss1 * 0.05) + (loss2 * 0.15) + (loss3 * 0.8)
            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            acc1 = f1_score(tensor2list(label1), tensor2list(pred1.argmax(1)), average='weighted')
            acc2 = f1_score(tensor2list(label2), tensor2list(pred2.argmax(1)), average='weighted')
            acc3 = f1_score(tensor2list(label3), tensor2list(pred3.argmax(1)), average='weighted')

            train_acc.append([acc1, acc2, acc3])
            train_loss.append([loss1.item(), loss2.item(), loss3.item()])

            mean_train_acc = np.mean(train_acc, axis=0)
            mean_train_loss = np.mean(train_loss, axis=0)

            tqdm_train.set_postfix({
                'Epoch': epoch,
                'Training Acc 1 ': mean_train_acc[0],
                'Training Acc 2 ': mean_train_acc[1],
                'Training Acc 3 ': mean_train_acc[2],
                'Training Loss 1 ': mean_train_loss[0],
                'Training Loss 2 ': mean_train_loss[1],
                'Training Loss 3 ': mean_train_loss[2],
            })

            data = {
                'training acc/1': acc1,
                'training acc/2': acc2,
                'training acc/3': acc3,
                'training loss/1': loss1.item(),
                'training loss/2': loss2.item(),
                'training loss/3': loss3.item(),
                'Learning rate' : self.optimizer.param_groups[0]["lr"]
            }
            logging(self.log_writter, data, epoch * len(self.train_loader) + batch)


    def validation(self, epoch):
        self.model.eval()

        val_acc, val_loss = [], []
        tqdm_valid = tqdm(self.val_loader)
        with torch.no_grad():
            # for batch, (img, txt, att_mask, label1, label2, label3) in enumerate(tqdm_valid):
            for batch, (img, label1, label2, label3) in enumerate(tqdm_valid, start=1):

                img = img.to(device)
                # txt = txt.to(device)
                # att_mask = att_mask.to(device)
                label1 = label1.to(device)
                label2 = label2.to(device)
                label3 = label3.to(device)

                pred1, pred2, pred3 = self.model(img)
                loss1 = self.criterion(pred1, label1)
                loss2 = self.criterion(pred2, label2)
                loss3 = self.criterion(pred3, label3)

                acc1 = f1_score(tensor2list(label1), tensor2list(pred1.argmax(1)), average='weighted')
                acc2 = f1_score(tensor2list(label2), tensor2list(pred2.argmax(1)), average='weighted')
                acc3 = f1_score(tensor2list(label3), tensor2list(pred3.argmax(1)), average='weighted')

                val_acc.append([acc1, acc2, acc3])
                val_loss.append([loss1.item(), loss2.item(), loss3.item()])

                mean_val_acc = np.mean(val_acc, axis=0)
                mean_val_loss = np.mean(val_loss, axis=0)

                tqdm_valid.set_postfix({
                    'Epoch': epoch,
                    'Valid Acc 1 ': mean_val_acc[0],
                    'Valid Acc 2 ': mean_val_acc[1],
                    'Valid Acc 3 ': mean_val_acc[2],
                    'Valid Loss 1 ': mean_val_loss[0],
                    'Valid Loss 2 ': mean_val_loss[1],
                    'Valid Loss 3 ': mean_val_loss[2],
                })

                data = {
                    'valid acc/1': acc1,
                    'valid acc/2': acc2,
                    'valid acc/3': acc3,
                    'valid loss/1': loss1.item(),
                    'valid loss/2': loss2.item(),
                    'valid loss/3': loss3.item(),
                }
                logging(self.log_writter, data, epoch * len(self.val_loader) + batch)

        self.model_save(epoch, mean_val_acc[2])

    def kfold_setup(self, model, criterion, optimizer, lr_scheduler, train_ind, valid_ind, img_set, label_set, transform, kfold):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler

        self.train_loader, self.val_loader = train_and_valid_dataload((img_set[train_ind], img_set[valid_ind]),
                                                                      (label_set.iloc[train_ind], label_set.iloc[valid_ind]),
                                                                      csv_path=self.args.CSV_PATH,
                                                                      transform=transform,
                                                                      batch_size=self.args.BATCH_SIZE)
        if kfold is not None :
            self.log_writter = SummaryWriter(os.path.join(self.args.LOG , str(kfold)))
            self.save_path = os.path.join(self.args.OUTPUT, str(kfold) + self.args.MODEL_NAME)


    def model_save(self, epoch, val_acc):
        if self.best_score < val_acc:
            self.best_score = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
            }, os.path.join(self.save_path, str(epoch) + 'E-val' + str(round(self.best_score, 4)) + '-' + self.args.SAVE_NAME + '.pth'))


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--BATCH_SIZE", type=int, default=16)
    parser.add_argument("--LEARNING_RATE", type=float, default=0.007)
    parser.add_argument("--EPOCHS", type=int, default=60)
    parser.add_argument("--IMG_MODEL_NAME", type=str, default='beit_base_patch16_384')
    parser.add_argument("--TXT_MODEL_NAME", type=str, default='klue/roberta-base')
    parser.add_argument("--MODEL_TYPE", type=str, default='transformer')
    parser.add_argument("--KFOLD", type=int, default=5)
    parser.add_argument("--RESIZE", type=int, default=384)
    parser.add_argument("--IMG_PATH", type=str, default="./data/image/train/*")
    parser.add_argument("--CSV_PATH", type=str, default="./data/train.csv")
    parser.add_argument("--LABEL_WEIGHT", type=str, default="./data/*_label_weight2.npy")
    parser.add_argument("--CLS_CONFIG", type=str, default='./config/cls_config.yaml')
    parser.add_argument("--LOSS_CONFIG", type=str, default='./config/loss_config.yaml')

    parser.add_argument("--OUTPUT", type=str, default='./ckpt/head_beit_bp16_384')
    parser.add_argument("--SAVE_NAME", type=str, default='head_beit_bp16_384')
    parser.add_argument("--LOG", type=str, default='./tensorboard/head_beit_bp16_384')

    parser.add_argument("--REUSE", type=bool, default=False)
    parser.add_argument("--CHECKPOINT", type=str, default='./ckpt/')

    parser.add_argument("--START_EPOCH", type=int, default=0)
    parser.add_argument("--AUG", type=bool, default=True)

    torch.autograd.set_detect_anomaly(True)

    args = parser.parse_args()
    cls_config = read_cfg(args.CLS_CONFIG)
    loss_config = read_cfg(args.LOSS_CONFIG)

    label_weight = read_label_weight(args.LABEL_WEIGHT)

    _args = vars(args)
    _args.update(cls_config)
    _args.update(loss_config)
    save_config(_args, args.OUTPUT)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # model = MultiTaskModel(img_model_name=args.IMG_MODEL_NAME,
    #                        txt_model_name=args.TXT_MODEL_NAME,
    #                        head1_cfg=cls_config['head1'],
    #                        head2_cfg=cls_config['head2'],
    #                        head3_cfg=cls_config['head3']).to("cuda")
    model = TransformerClassifier(model_name=args.IMG_MODEL_NAME,
                                  model_type=args.MODEL_TYPE,
                                  head_config=cls_config).to("cuda")
    criterion = FocalLoss(**loss_config['focal'])
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.LEARNING_RATE)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

    if args.KFOLD == 0 :
        print("***** NO KFOLD *****")
        trainer = Trainer(model, criterion, optimizer, lr_scheduler, device, args)
        trainer.run()

    elif args.KFOLD > 0 :
        kfold = StratifiedKFold(n_splits=args.KFOLD, random_state=99, shuffle=True)
        trainer = Trainer(model, criterion, optimizer, lr_scheduler, device, args)

        img_set, label_set, transform = image_label_dataset(args.CSV_PATH, resize=args.RESIZE, training=True)

        for k, (train_ind, valid_ind) in enumerate(kfold.split(img_set, label_set['cat3'])):
            trainer.kfold_setup(model, criterion, optimizer, lr_scheduler, train_ind, valid_ind, np.array(img_set), label_set, transform, kfold=None)
            trainer.run()
            break