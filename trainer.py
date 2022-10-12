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
from models import BackBone, ClassifierHead1, ClassifierHead2, ClassifierHead3
from loss_fn import FocalLoss

class Trainer() :
    def __init__ (self, backbone, head1, head2, head3,
                  criterion1, criterion2, criterion3, optimizer,
                  device, args) :
        self.backbone = backbone
        self.head1 = head1
        self.head2 = head2
        self.head3 = head3

        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.criterion3 = criterion3

        self.optimizer = optimizer

        self.train_loader, self.val_loader = self.get_dataloader(args.CSV_PATH, args.IMG_PATH, args.BATCH_SIZE,
                                                                 args.RESIZE)
        # self.masking_generator = CATEGORY_MASKING(path=args.CSV_PATH)
        self.cls_weight_generator = CLS_WEIGHT(csv_path=args.CSV_PATH)

        self.log_writter = SummaryWriter(args.LOG)
        self.save_path = args.OUTPUT
        self.args = args
        self.best_score = 0
        self.device = device
        self.APPLY_MIXUP = False

    def run(self):
        if self.args.REUSE :
            self.backbone, self.head1, self.head2, self.head3, self.optimizer, self.args.START_EPOCH = weight_load(self.backbone,
                                                                                                        self.head1,
                                                                                                        self.head2,
                                                                                                        self.head3,
                                                                                                        self.optimizer,
                                                                                                        self.args.CHECKPOINT)
            
        for epoch in range(self.args.START_EPOCH + 1, self.args.EPOCHS + 1) :

            # training
            self.training(epoch)

            # validation
            # self.validation(epoch)


    def training(self, epoch):
        self.backbone.train()
        self.head1.train()
        self.head2.train()
        self.head3.train()

        tqdm_train = tqdm(self.train_loader)
        train_acc, train_loss = [], []
        # for batch, (img, label1, (label2, mask2), (label3, mask3)) in enumerate(tqdm_train, start=1):
        for batch, (img, label1, label2, label3) in enumerate(tqdm_train, start=1):
            self.optimizer.zero_grad()

            img = img.to(self.device)
            label1 = label1.to(self.device)
            label2 = label2.to(self.device)
            label3 = label3.to(self.device)
            # mask2 = mask2.to(self.device)
            # mask3 = mask3.to(self.device)

            feature_map = self.backbone(img)

            pred1 = self.head1(feature_map) # [batch, 6]


            mask2 = self.cls_weight_generator(pred1, num=18, category=1)

            # print(pred1)
            # print(mask2)

            pred2 = self.head2(feature_map, mask2, mode='weight') # [batch, 18]

            mask3 = self.cls_weight_generator(pred2, num=128, category=2)
            pred3 = self.head3(feature_map, mask3, mode='weight') # # [batch, 128]

            loss1 = self.criterion1(pred1, label1)
            loss2 = self.criterion2(pred2, label2)
            loss3 = self.criterion3(pred3, label3)
            loss = loss1 + loss2 + loss3
            loss.backward()

            self.optimizer.step()

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
            }
            logging(self.log_writter, data, epoch * len(self.train_loader) + batch)
        self.model_save(epoch, mean_train_acc[2])

    def validation(self, epoch):
        self.backbone.eval()
        self.head1.eval()
        self.head2.eval()
        self.head3.eval()

        # label_enc = LABEL_ENCODER(self.args.CSV_PATH)
        # cat2_ig_enc = label_enc.cat2_label_index_encoder()
        # cat3_ig_enc = label_enc.cat3_label_index_encoder()

        val_acc, val_loss = [], []
        tqdm_valid = tqdm(self.val_loader)
        with torch.no_grad():
            # for batch, (img, label1, (label2, mask2), (label3, mask3)) in enumerate(tqdm_valid):
            for batch, (img, label1, label2, label3,) in enumerate(tqdm_valid):
                img = img.to(self.device)
                label1 = label1.to(self.device)
                label2 = label2.to(self.device)
                label3 = label3.to(self.device)
                # mask2 = mask2.to(self.device)
                # mask3 = mask3.to(self.device)

                feature_map = self.backbone(img)
                pred1 = self.head1(feature_map)

                mask2 = self.cls_weight_generator(pred1, num=18, category=1)
                pred2 = self.head2(feature_map, mask2, mode='weight')  # [batch, 18]

                mask3 = self.cls_weight_generator(pred2, num=128, category=2)
                pred3 = self.head3(feature_map, mask3, mode='weight')  # # [batch, 128]

                loss1 = self.criterion1(pred1, label1)
                loss2 = self.criterion2(pred2, label2)
                loss3 = self.criterion3(pred3, label3)

                acc1 = f1_score(tensor2list(label1), tensor2list(pred1.argmax(1)), average='weighted')
                acc2 = f1_score(tensor2list(label2), tensor2list(pred2.argmax(1)), average='weighted')
                acc3 = f1_score(tensor2list(label3), tensor2list(pred3.argmax(1)), average='weighted')

                val_acc.append([acc1, acc2, acc3])
                val_loss.append([loss1.item(), loss2.item(), loss3.item()])

                mean_val_acc = np.mean(val_acc, axis=0)
                mean_val_loss = np.mean(val_loss, axis=0)

                tqdm_valid.set_postfix({
                    'Epoch' : epoch,
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

    def kfold_setup(self, backbone, head1, head2, head3, criterion, optimizer, train_ind, valid_ind, kfold):
        self.backbone = backbone
        self.head1 = head1
        self.head2 = head2
        self.head3 = head3
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader, self.val_loader = train_and_valid_dataload((self.img_set[train_ind], self.img_set[valid_ind]),
                                                                      (self.label_set[train_ind], self.label_set[valid_ind]),
                                                                      self.transform,
                                                                      batch_size=self.args.BATCH_SIZE)
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
                "backbone_state_dict": self.backbone.state_dict(),
                "head1_state_dict": self.head1.state_dict(),
                "head2_state_dict": self.head2.state_dict(),
                "head3_state_dict": self.head3.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
            }, os.path.join(self.save_path, str(epoch) + 'E-val' + str(self.best_score) + '-' + self.args.MODEL_NAME + '.pth'))


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--BATCH_SIZE", type=int, default=32)
    parser.add_argument("--LEARNING_RATE", type=float, default=0.001)
    parser.add_argument("--EPOCHS", type=int, default=60)
    parser.add_argument("--MODEL_NAME", type=str, default='efficientnetv2_rw_m')
    parser.add_argument("--MODEL_TYPE", type=str, default='cnn')
    parser.add_argument("--KFOLD", type=int, default=0)
    parser.add_argument("--RESIZE", type=int, default=224)
    parser.add_argument("--IMG_PATH", type=str, default="./data/image/train/*")
    parser.add_argument("--CSV_PATH", type=str, default="./data/train.csv")
    parser.add_argument("--LABEL_WEIGHT", type=str, default="./data/*_label_weight2.npy")
    parser.add_argument("--CLS_CONFIG", type=str, default='./config/cls_config.yaml')
    parser.add_argument("--LOSS_CONFIG", type=str, default='./config/loss_config.yaml')

    parser.add_argument("--OUTPUT", type=str, default='./ckpt/newlr0.01_fullData_maskingWeight_efficientnetv2_m')
    parser.add_argument("--LOG", type=str, default='./tensorboard/newlr0.01_fullData_maskingWeight_efficientnetv2_m')

    parser.add_argument("--REUSE", type=bool, default=True)
    parser.add_argument("--CHECKPOINT", type=str, default='./ckpt/each_head/lossWeight-auged-effiv2s/28E-val0.9198814332761365-efficientnetv2_rw_s.pth')

    parser.add_argument("--START_EPOCH", type=int, default=0)

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

    backbone = BackBone(model_name=args.MODEL_NAME, model_type=args.MODEL_TYPE).to('cuda')
    head1 = ClassifierHead1(**cls_config['head1']).to("cuda")
    head2 = ClassifierHead2(**cls_config['head2']).to("cuda")
    head3 = ClassifierHead3(**cls_config['head3']).to("cuda")

    criterion1 = FocalLoss(**loss_config['focal'])
    criterion2 = FocalLoss(**loss_config['focal'])
    criterion3 = FocalLoss(**loss_config['focal'])

    optimizer = torch.optim.AdamW([{'params' : backbone.parameters()},
                                 {'params' : head1.parameters()},
                                 {'params' : head2.parameters()},
                                 {'params' : head3.parameters()}],
                                 lr=args.LEARNING_RATE)

    if args.KFOLD == 0 :
        print("***** NO KFOLD *****")
        trainer = Trainer(backbone, head1, head2, head3,
                          criterion1, criterion2, criterion3, optimizer,
                          device, args)
        trainer.run()

    elif args.KFOLD > 0 :
        kfold = StratifiedKFold(n_splits=args.KFOLD, shuffle=True)
        trainer = Trainer(backbone, head1, head2, head3, criterion, optimizer, device, args)
        for k, (train_ind, valid_ind) in enumerate(kfold.split(trainer.img_set, trainer.label_set)) :
            trainer.kfold_setup(backbone, head1, head2, head3, criterion, optimizer, train_ind, valid_ind, k)
            trainer.run()