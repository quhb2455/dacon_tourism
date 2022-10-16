import torch
import os
import argparse
from tqdm import tqdm

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

from torch.utils.tensorboard import SummaryWriter

from datasets import *
from utils import *
from models import CNN, BackBone, ClassifierHead1, ClassifierHead2, ClassifierHead3
from loss_fn import FocalLoss

class Trainer() :
    def __init__ (self, mask1_cnn, mask2_cnn, cnn, criterion, optimizer, device, args) :
        # setup for masking
        self.mask1_cnn = mask1_cnn
        self.mask2_cnn = mask2_cnn
        self.cls_weight_generator = CLS_WEIGHT(csv_path=args.CSV_PATH)

        # train cnn
        self.model = cnn
        self.criterion = criterion
        self.optimizer = optimizer

        self.train_loader, self.val_loader = self.get_dataloader(args.CSV_PATH, args.IMG_PATH, args.BATCH_SIZE,
                                                                 args.RESIZE)

        self.log_writter = SummaryWriter(args.LOG)
        self.save_path = args.OUTPUT
        self.APPLY_MIXUP = args.MIXUP
        self.args = args

        self.best_score = 0
        self.early_stop_cnt = 0
        self.device = device


    def run(self):
        if self.args.REUSE :
            self.model, self.optimizer, self.args.START_EPOCH = single_model_weight_load(self.model,
                                                                                        self.optimizer,
                                                                                        self.args.CHECKPOINT)
            
        for epoch in range(self.args.START_EPOCH + 1, self.args.EPOCHS + 1) :

            # training
            self.training(epoch)

            # validation
            self.validation(epoch)

            if self.early_stop_cnt == self.args.EARLY_STOP :
                print(" ======= Early Stopp =======")
                break


    def training(self, epoch):
        self.model.train()

        tqdm_train = tqdm(self.train_loader)
        train_acc, train_loss = [], []
        for batch, (img, label1, label2, label3) in enumerate(tqdm_train, start=1):
            self.optimizer.zero_grad()

            img = img.to(self.device)
            if self.args.TRAINING_HEAD == 'head1':
                label = label1.to(self.device)
            elif self.args.TRAINING_HEAD == 'head2':
                label = label2.to(self.device)
            elif self.args.TRAINING_HEAD == 'head3':
                label = label3.to(self.device)

            if self.APPLY_MIXUP :
                img, lam, label_a, label_b = mixup(img, label)
                pred = self.model(img)
                loss = lam * self.criterion(pred, label_a) + (1 - lam) * self.criterion(pred, label_b)
            else:
                # create mask
                mask1_pred = self.mask1_cnn(img)
                mask1 = self.cls_weight_generator(mask1_pred, num=18, category=1)
                mask2_pred = self.mask2_cnn(img, mask=mask1)
                mask = self.cls_weight_generator(mask2_pred, num=128, category=2)

                # prediction with mask
                pred = self.model(img, mask=mask)
                loss = self.criterion(pred, label)

            loss.backward()
            self.optimizer.step()

            acc = f1_score(tensor2list(label), tensor2list(pred.argmax(1)), average='weighted')

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
                'training acc': acc,
                'training loss': loss.item()
            }
            logging(self.log_writter, data, epoch * len(self.train_loader) + batch)

    def validation(self, epoch):
        self.model.eval()

        val_acc, val_loss = [], []
        tqdm_valid = tqdm(self.val_loader)
        with torch.no_grad():
            # for batch, (img, label1, (label2, mask2), (label3, mask3)) in enumerate(tqdm_valid):
            for batch, (img, label1, label2, label3,) in enumerate(tqdm_valid):
                img = img.to(self.device)
                if self.args.TRAINING_HEAD == 'head1':
                    label = label1.to(self.device)
                elif self.args.TRAINING_HEAD == 'head2':
                    label = label2.to(self.device)
                elif self.args.TRAINING_HEAD == 'head3':
                    label = label3.to(self.device)

                # create mask
                mask1_pred = self.mask1_cnn(img)
                mask1 = self.cls_weight_generator(mask1_pred, num=18, category=1)
                mask2_pred = self.mask2_cnn(img, mask=mask1)
                mask = self.cls_weight_generator(mask2_pred, num=128, category=2)

                # prediction with mask
                pred = self.model(img, mask=mask)
                loss = self.criterion(pred, label)

                acc = f1_score(tensor2list(label), tensor2list(pred.argmax(1)), average='weighted')

                val_acc.append(acc)
                val_loss.append(loss.item())

                mean_val_acc = np.mean(val_acc, axis=0)
                mean_val_loss = np.mean(val_loss, axis=0)

                tqdm_valid.set_postfix({
                    'Epoch' : epoch,
                    'Valid Acc ': mean_val_acc,
                    'Valid Loss  ': mean_val_loss,
                })

                data = {
                    'valid acc': acc,
                    'valid loss': loss.item(),
                }
                logging(self.log_writter, data, epoch * len(self.val_loader) + batch)

        self.model_save(epoch, mean_val_acc)

    def kfold_setup(self, model, criterion, optimizer, train_ind, valid_ind, img_set, label_set, kfold):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader, self.val_loader = train_and_valid_dataload((img_set[train_ind], img_set[valid_ind]),
                                                                      (label_set.iloc[train_ind], label_set.iloc[valid_ind]),
                                                                      csv_path=self.args.CSV_PATH,
                                                                      transform=self.transform,
                                                                      batch_size=self.args.BATCH_SIZE)
        if kfold is not None :
            self.log_writter = SummaryWriter(os.path.join(self.args.LOG , str(kfold)))
            self.save_path = os.path.join(self.args.OUTPUT, str(kfold) + self.args.MODEL_NAME)


    def get_dataloader(self, csv_path, img_path, batch_size, resize=224):
        self.img_set, self.label_set, self.transform = image_label_dataset(csv_path,
                                                                           img_path,
                                                                           div=1,
                                                                           resize=resize,
                                                                           training=True)
        return train_and_valid_dataload(self.img_set, self.label_set, csv_path, self.transform, batch_size=batch_size)


    def model_save(self, epoch, val_acc):
        if self.best_score < val_acc:
            self.best_score = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
            }, os.path.join(self.save_path, str(epoch) + 'E-val' + str(round(self.best_score, 4)) + '-' + self.args.MODEL_NAME + '.pth'))
            self.early_stop_cnt = 0
        else :
            self.early_stop_cnt += 1
            print(f"Early stopping count : {self.early_stop_cnt}/{self.args.EARLY_STOP}")




if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--BATCH_SIZE", type=int, default=16)
    parser.add_argument("--LEARNING_RATE", type=float, default=0.001)
    parser.add_argument("--EPOCHS", type=int, default=60)
    parser.add_argument("--EARLY_STOP", type=int, default=6)
    parser.add_argument("--MODEL_NAME", type=str, default='efficientnetv2_rw_s')
    parser.add_argument("--MODEL_TYPE", type=str, default='cnn')

    parser.add_argument("--TRAINING_HEAD", type=str, default='head3')

    parser.add_argument("--KFOLD", type=int, default=4)
    parser.add_argument("--RESIZE", type=int, default=224)
    parser.add_argument("--IMG_PATH", type=str, default="./data/image/train/*")
    parser.add_argument("--CSV_PATH", type=str, default="./data/aug_train.csv")
    parser.add_argument("--LABEL_WEIGHT", type=str, default="./data/*_label_weight2.npy")
    parser.add_argument("--CLS_CONFIG", type=str, default='./config/cls_config.yaml')
    parser.add_argument("--LOSS_CONFIG", type=str, default='./config/loss_config.yaml')

    parser.add_argument("--OUTPUT", type=str, default='./ckpt/each_head/3/head1-2mask-lossWeight-auged-effiv2s')
    parser.add_argument("--LOG", type=str, default='./tensorboard/each_head/3/head1-2mask-lossWeight-auged-effiv2s')

    parser.add_argument("--REUSE", type=bool, default=True)
    # parser.add_argument("--CHECKPOINT", type=str, default='./ckpt/each_head/3/head1-2mask-lossWeight-auged-effiv2s/.pth')
    parser.add_argument("--CHECKPOINT", type=str,
                        default='./ckpt/each_head/3/head1-2mask-lossWeight-auged-effiv2s/26E-val0.6251-efficientnetv2_rw_s.pth')
    parser.add_argument("--START_EPOCH", type=int, default=0)

    parser.add_argument("--MIXUP", type=bool, default=False)
    torch.autograd.set_detect_anomaly(True)

    args = parser.parse_args()
    cls_config = read_cfg(args.CLS_CONFIG, args.TRAINING_HEAD)
    loss_config = read_cfg(args.LOSS_CONFIG)

    label_weight = read_label_weight(args.LABEL_WEIGHT, cat=args.TRAINING_HEAD)

    _args = vars(args)
    _args.update(cls_config)
    _args.update(loss_config)
    save_config(_args, args.OUTPUT)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # head 1 setup
    cls1_config = read_cfg(args.CLS_CONFIG, 'head1')
    head1_cnn = CNN(model_name=args.MODEL_NAME,
                    model_type=args.MODEL_TYPE,
                    head_config=cls1_config,
                    head_name='head1').to("cuda")
    head1_cnn = single_model_weight_load(head1_cnn,
                                         None,
                                         './ckpt/each_head/1/lossWeight-auged-effiv2s/29E-val0.9343-efficientnetv2_rw_s.pth',
                                         training=False)
    head1_cnn.eval()

    # head 2 setup
    cls2_config = read_cfg(args.CLS_CONFIG, 'head2')
    head2_cnn = CNN(model_name=args.MODEL_NAME,
                    model_type=args.MODEL_TYPE,
                    head_config=cls2_config,
                    head_name='head2',
                    mode='weight').to("cuda")
    head2_cnn = single_model_weight_load(head2_cnn,
                                         None,
                                         './ckpt/each_head/2/head1mask-lossWeight-auged-effiv2s/26E-val0.9051-efficientnetv2_rw_s.pth',
                                         training=False)
    head2_cnn.eval()

    # head3
    cnn = CNN(model_name=args.MODEL_NAME,
              model_type=args.MODEL_TYPE,
              head_config=cls_config,
              head_name=args.TRAINING_HEAD,
              mode='weight').to("cuda")
    criterion = FocalLoss(**loss_config['focal'])
    # criterion = FocalLoss(**loss_config['focal'], weight=label_weight)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=args.LEARNING_RATE)


    kfold = StratifiedKFold(n_splits=args.KFOLD, shuffle=True)
    trainer = Trainer(head1_cnn, head2_cnn, cnn, criterion, optimizer, device, args)
    img_set = np.array(trainer.img_set[0])
    label_set = trainer.label_set[0]
    if args.TRAINING_HEAD == 'head1':
        _label_set = label_set['cat1']
    elif args.TRAINING_HEAD == 'head2':
        _label_set = label_set['cat2']
    elif args.TRAINING_HEAD == 'head3':
        _label_set = label_set['cat3']

    for k, (train_ind, valid_ind) in enumerate(kfold.split(img_set, _label_set)):
        trainer.kfold_setup(cnn, criterion, optimizer, train_ind, valid_ind, img_set, label_set, kfold=None)
        trainer.run()
        break