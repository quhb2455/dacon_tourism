import torch
import os
import argparse
from tqdm import tqdm

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from datasets import CustomDataset, train_and_valid_dataload, image_label_dataset
from utils import *
from models import *
from loss_fn import FocalLoss


class Trainer():
    def __init__(self, model, criterion, optimizer, device, args):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.log_writter = SummaryWriter(args.LOG)
        self.save_path = args.OUTPUT
        self.args = args
        self.best_score = 0
        self.device = device
        self.APPLY_MIXUP = False

    def run(self):
        if self.args.REUSE:
            self.model, self.optimizer, self.args.START_EPOCH = weight_load(self.model, self.optimizer,
                                                                            self.args.CHECKPOINT)

        for epoch in range(self.args.START_EPOCH + 1, self.args.EPOCHS + 1):
            # training
            self.training(epoch)

            # validation
            self.validation(epoch)

    def training(self, epoch):
        self.model.train()

        tqdm_train = tqdm(self.train_loader)
        train_acc, train_loss = [], []
        # for batch, (img, label1, (label2, mask2), (label3, mask3)) in enumerate(tqdm_train, start=1):
        for batch, (img, label1, label2, label3) in enumerate(tqdm_train, start=1):
            self.optimizer.zero_grad()

            img = img.to(device)
            label3 = label3.to(device)

            if self.args.AUG:
                p = np.random.rand(1)

                # mixup
                if p < 0.5 :
                    img, lam, label_a, label_b = mixup(img, label3)
                    pred = self.model(img)
                    loss = lam * self.criterion(pred, label_a) + (1 - lam) * self.criterion(pred, label_b)

                # cutmix
                else :
                    img, lam, label_a, label_b = cutmix(img, label3)
                    pred = self.model(img)
                    loss = lam * self.criterion(pred, label_a) + (1 - lam) * self.criterion(pred, label_b)

            else:
                pred = self.model(img)
                loss = self.criterion(pred, label3)

            loss.backward()
            self.optimizer.step()

            acc = f1_score(tensor2list(label3), tensor2list(pred.argmax(1)), average='weighted')

            train_acc.append(acc)
            train_loss.append(loss.item())

            mean_train_acc = np.mean(train_acc, axis=0)
            mean_train_loss = np.mean(train_loss, axis=0)

            tqdm_train.set_postfix({
                'Epoch': epoch,
                'Training Acc ': mean_train_acc,
                'Training Loss ': mean_train_loss
            })

            data = {
                'training acc ': acc,
                'training loss': loss.item()
            }
            logging(self.log_writter, data, epoch * len(self.train_loader) + batch)

    def validation(self, epoch):
        self.model.eval()

        val_acc, val_loss = [], []
        tqdm_valid = tqdm(self.val_loader)
        with torch.no_grad():
            for batch, (img, label1, label2, label3) in enumerate(tqdm_valid, start=1):
                img = img.to(device)
                label3 = label3.to(device)

                pred = self.model(img)
                loss = self.criterion(pred, label3)

                acc = f1_score(tensor2list(label3), tensor2list(pred.argmax(1)), average='weighted')

                val_acc.append(acc)
                val_loss.append(loss.item())

                mean_val_acc = np.mean(val_acc, axis=0)
                mean_val_loss = np.mean(val_loss, axis=0)

                tqdm_valid.set_postfix({
                    'Epoch': epoch,
                    'Valid Acc ': mean_val_acc,
                    'Valid Loss ': mean_val_loss
                })

                data = {
                    'valid acc': acc,
                    'valid loss': loss.item()
                }
                logging(self.log_writter, data, epoch * len(self.val_loader) + batch)

        self.model_save(epoch, mean_val_acc)

    def kfold_setup(self, model, criterion, optimizer, train_ind, valid_ind, img_set, label_set, transform, kfold):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader, self.val_loader = train_and_valid_dataload((img_set[train_ind], img_set[valid_ind]),
                                                                      (label_set.iloc[train_ind],
                                                                       label_set.iloc[valid_ind]),
                                                                      csv_path=self.args.CSV_PATH,
                                                                      transform=transform,
                                                                      batch_size=self.args.BATCH_SIZE)
        if kfold is not None:
            self.log_writter = SummaryWriter(os.path.join(self.args.LOG, str(kfold)))
            self.save_path = os.path.join(self.args.OUTPUT, str(kfold) + self.args.MODEL_NAME)

    def model_save(self, epoch, val_acc):
        if self.best_score < val_acc:
            self.best_score = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
            }, os.path.join(self.save_path,
                            str(epoch) + 'E-val' + str(round(self.best_score, 4)) + '-' + self.args.SAVE_NAME + '.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--BATCH_SIZE", type=int, default=16)
    parser.add_argument("--LEARNING_RATE", type=float, default=0.01)
    parser.add_argument("--EPOCHS", type=int, default=60)
    parser.add_argument("--MODEL_NAME", type=str, default='beit_base_patch16_384')
    parser.add_argument("--MODEL_TYPE", type=str, default='cnn')
    parser.add_argument("--KFOLD", type=int, default=5)
    parser.add_argument("--RESIZE", type=int, default=384)

    parser.add_argument("--IMG_PATH", type=str, default="./data/image/train/*")
    parser.add_argument("--CSV_PATH", type=str, default="./data/train.csv")

    parser.add_argument("--LABEL_WEIGHT", type=str, default="./data/*_label_weight2.npy")
    parser.add_argument("--CLS_CONFIG", type=str, default='./config/cls_config.yaml')
    parser.add_argument("--LOSS_CONFIG", type=str, default='./config/loss_config.yaml')

    parser.add_argument("--OUTPUT", type=str, default='./ckpt/beit_bp16_384')
    parser.add_argument("--SAVE_NAME", type=str, default='beit_bp16_384')
    parser.add_argument("--LOG", type=str, default='./tensorboard/beit_bp16_384')

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
    model = TransformerMODEL(model_name=args.MODEL_NAME).to("cuda")
    criterion = FocalLoss(**loss_config['focal'])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.LEARNING_RATE)

    if args.KFOLD == 0:
        print("***** NO KFOLD *****")
        trainer = Trainer(model, criterion, optimizer, device, args)
        trainer.run()

    elif args.KFOLD > 0:

        kfold = StratifiedKFold(n_splits=args.KFOLD, random_state=99, shuffle=True)
        trainer = Trainer(model, criterion, optimizer, device, args)

        img_set, label_set, transform = image_label_dataset(args.CSV_PATH, resize=args.RESIZE, training=True)

        for k, (train_ind, valid_ind) in enumerate(kfold.split(img_set, label_set['cat3'])):

            trainer.kfold_setup(model, criterion, optimizer, train_ind, valid_ind, np.array(img_set), label_set,
                                transform, kfold=None)
            trainer.run()
            break