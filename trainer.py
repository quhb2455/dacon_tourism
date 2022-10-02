import torch
import os
import argparse
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

from datasets import *
from utils import *
from models import MultiTaskModel
from loss_fn import FocalLoss

class Trainer() :
    def __init__ (self, model, optimizer, device, args) :
        self.model = model
        self.optimizer = optimizer

        self.train_loader ,self.val_loader = self.get_dataloader(args.CSV_PATH, args.IMG_PATH, args.BATCH_SIZE)
        
        self.log_writter = SummaryWriter(args.LOG)
        self.save_path = args.OUTPUT
        self.args = args
        self.best_score = 0
        self.device = device
        self.APPLY_MIXUP = False

    def run(self):
        if self.args.REUSE :
            self.model, self.optimizer, self.args.START_EPOCH = weight_load(self.model,
                                                                            self.optimizer,
                                                                            self.args.CHECKPOINT)
            
        for epoch in range(self.args.START_EPOCH + 1, self.args.EPOCHS + 1) :

            # training
            self.training(epoch)

            # validation
            self.validation(epoch)


    def training(self, epoch):
        tqdm_train = tqdm(self.train_loader)
        train_acc, train_loss = [], []
        for batch, (img, label) in enumerate(tqdm_train, start=1):
            self.model.train()
            self.optimizer.zero_grad()

            img = img.to(self.device)
            label = label.to(self.device)

            loss.backward()
            self.optimizer.step()

            acc = score(label, output)
            train_acc.append(acc)
            train_loss.append(loss.item())

            tqdm_train.set_postfix({
                'Epoch': epoch,
                'Training Acc': np.mean(train_acc),
                'Training Loss': np.mean(train_loss)
            })

            data = {
                'training loss': loss.item(),
                'training acc': acc
            }
            logging(self.log_writter, data, epoch * len(self.train_loader) + batch)

    def validation(self, epoch):
        self.model.eval()
        true_labels = []
        model_preds = []
        val_loss = []
        tqdm_valid = tqdm(self.val_loader)
        with torch.no_grad():
            for batch, (img, label) in enumerate(tqdm_valid):
                img = img.to(self.device)
                label = label.to(self.device)

                model_pred = self.model(img)

                val_loss.append(loss.item())

                model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
                true_labels += label.detach().cpu().numpy().tolist()

                mean_loss = np.mean(val_loss)
                acc = accuracy_score(true_labels, model_preds)

                tqdm_valid.set_postfix({
                    'Epoch' : epoch,
                    'Valid Acc': acc,
                    'Valid Loss': mean_loss
                })

                data = {
                    'validation loss': mean_loss,
                    'validation acc': acc
                }
                logging(self.log_writter, data, epoch * len(self.val_loader) + batch)
                
        self.model_save(epoch, acc)

    def kfold_setup(self, model, optimizer, train_ind, valid_ind, kfold):
        self.model = model
        self.optimizer = optimizer
        self.train_loader, self.val_loader = train_and_valid_dataload((self.img_set[train_ind], self.img_set[valid_ind]),
                                                                      (self.label_set[train_ind], self.label_set[valid_ind]),
                                                                      self.transform,
                                                                      batch_size=self.args.BATCH_SIZE)
        self.log_writter = SummaryWriter(os.path.join(self.args.LOG , str(kfold)))
        self.save_path = os.path.join(self.args.OUTPUT, str(kfold) + self.args.MODEL_NAME)


    def get_dataloader(self, csv_path, img_path, batch_size):
        self.img_set, self.label_set, self.transform = image_label_dataset(csv_path,
                                                                           img_path,
                                                                           div=0.8,
                                                                           resize=224,
                                                                           training=True)
        return train_and_valid_dataload(self.img_set, self.label_set, csv_path, self.transform, batch_size=batch_size)


    def model_save(self, epoch, val_acc):
        if self.best_score < val_acc:
            self.best_score = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
            }, os.path.join(self.save_path, str(epoch) + 'E-val' + str(self.best_score) + '-' + self.args.MODEL_NAME + '.pth'))


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--BATCH_SIZE", type=int, default=64)
    parser.add_argument("--LEARNING_RATE", type=float, default=1e-4)
    parser.add_argument("--EPOCHS", type=int, default=70)
    parser.add_argument("--CTL_STEP", nargs="+", type=int, default=[36, 61])
    parser.add_argument("--FOCAL_GAMMA", type=int, default=2)
    parser.add_argument("--FOCAL_ALPHA", type=int, default=2)
    parser.add_argument("--MODEL_NAME", type=str, default='swin_base_patch4_window7_224_in22k')
    parser.add_argument("--KFOLD", type=int, default=0)

    parser.add_argument("--IMG_PATH", type=str, default="./data/img/224img_train/*")
    parser.add_argument("--CSV_PATH", type=str, default="./data/train.csv")

    parser.add_argument("--OUTPUT", type=str, default='./ckpt')
    parser.add_argument("--LOG", type=str, default='./tensorboard/PCA_img/test')

    parser.add_argument("--REUSE", type=bool, default=True)
    parser.add_argument("--CHECKPOINT", type=str, default='./ckpt/3E-val0.8645-efficientnet_b0.pth')

    parser.add_argument("--START_EPOCH", type=int, default=0)


    args = parser.parse_args()
    os.makedirs(args.OUTPUT, exist_ok=True)

    cls_config = read_cfg('./config/cls_config.yaml')
    loss_config = read_cfg('./config/loss_config.yaml')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = MultiTaskModel(model_name=args.MODEL_NAME,
                           head1_cfg=cls_config['head1'],
                           head2_cfg=cls_config['head2'],
                           head3_cfg=cls_config['head3'],
                           loss_cfg=loss_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.LEARNING_RATE)

    if args.KFOLD == 0 :
        trainer = Trainer(model, optimizer, device, args)
        trainer.run()

    elif args.KFOLD > 0 :
        kfold = StratifiedKFold(n_splits=args.KFOLD, shuffle=True)
        trainer = Trainer(model, optimizer, device, args)
        for k, (train_ind, valid_ind) in enumerate(kfold.split(trainer.img_set, trainer.label_set)) :
            trainer.kfold_setup(model, optimizer, train_ind, valid_ind, k)
            trainer.run()