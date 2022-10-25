from data_utils import createDataLoader
from transformers.optimization import get_cosine_schedule_with_warmup
import options
import torch
import utils
from torch import nn
from models import TourismModel
from tqdm import tqdm
import numpy as np
import os
from transformers import BertModel, AutoModel
import inference

def train(model, train_loader, val_loader, device):
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = options.LEARNING_RATE)
    total_steps = len(train_loader) * options.EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(total_steps*0.1),
                    num_training_steps=total_steps
                )

    try:
        model, e = utils.load_model(model, utils.latest_checkpoint_path(options.MODEL_PATH), device)
        e += 1
    except:
        print('model load fail!')
        e = 0
        model = model.to(device)
    #print(model)
    criterion = nn.CrossEntropyLoss().to(device)
    best_score = 0
    best_model = None
    print('Train Start!')
    for epoch in range(e, options.EPOCHS+1):
        model.train()
        train_loss = []
        model_preds = [[] for _ in range(3)]
        true_labels = [[] for _ in range(3)]
        for img, text, attention_mask, label1, label2, label3, raw_text in tqdm(iter(train_loader)):
            label = []
            losses = []
            img = img.float().to(device)
            text = text.to(device)
            attention_mask = attention_mask.to(device)
            optimizer.zero_grad()
            pred1, pred2, pred3 = model(img, text, attention_mask)

            for l in [label1, label2, label3]:
                label.append(l.to(device))

            preds = [pred1, pred2, pred3]

            for p, l in zip(preds, label):
                losses.append(criterion(p, l))

            loss = losses[0] * 0.05 + losses[1] * 0.1 + losses[2] * 0.85

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())

            model_preds[0] += pred1.argmax(1).detach().cpu().numpy().tolist()
            model_preds[1] += pred2.argmax(1).detach().cpu().numpy().tolist()
            model_preds[2] += pred3.argmax(1).detach().cpu().numpy().tolist()
            true_labels[0] += label1.detach().cpu().numpy().tolist()
            true_labels[1] += label2.detach().cpu().numpy().tolist()
            true_labels[2] += label3.detach().cpu().numpy().tolist()

        train_score = utils.score_function(true_labels, model_preds)
        tr_loss = np.mean(train_loss)

        val_loss, val_score = validation(model, criterion, val_loader, device)

        print(f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}] Val Loss : [{val_loss:.5f}] Train Score : {train_score[2]:.5f} Val Score : [{val_score[0]:.5f}, {val_score[1]:.5f}, {val_score[2]:.5f}]')
        utils.save_log(epoch, tr_loss, val_loss, train_score, val_score)
        if epoch % options.SAVE_INTERVAL == 0:
            save_path = os.path.join(options.MODEL_PATH, 'checkpoint_' + str(epoch) + '.pt')
            utils.save_model(model, save_path, epoch)
            save_name = 'auto_roberta-s_E' + str(epoch) + '_lr3e5_adamw_fullPreprocessing_PosExtractor_S20_infer.csv'
            inference.infer_call(model, device, save_name)
        if best_score < val_score[2]:
            best_score = val_score[2]
            best_model = model

    return best_model

def validation(model, criterion, val_loader, device):
    model.eval()

    model_preds = [[] for _ in range(3)]
    true_labels = [[] for _ in range(3)]

    val_loss = []

    loss_w = [0.05, 0.15, 0.8]
    with torch.no_grad():
        for img, text, attention_mask, label1, label2, label3, raw_text in tqdm(iter(val_loader)):
            img = img.float().to(device)
            text = text.to(device)
            attention_mask = attention_mask.to(device)

            pred1, pred2, pred3 = model(img, text, attention_mask)

            label1 = label1.to(device)
            label2 = label2.to(device)
            label3 = label3.to(device)
            loss1 = criterion(pred1, label1)
            loss2 = criterion(pred2, label2)
            loss3 = criterion(pred3, label3)
            loss = loss1 * 0.05 + loss2 * 0.1 + loss3 * 0.85

            val_loss.append(loss.item())
            model_preds[0] += pred1.argmax(1).detach().cpu().numpy().tolist()
            model_preds[1] += pred2.argmax(1).detach().cpu().numpy().tolist()
            model_preds[2] += pred3.argmax(1).detach().cpu().numpy().tolist()
            true_labels[0] += label1.detach().cpu().numpy().tolist()
            true_labels[1] += label2.detach().cpu().numpy().tolist()
            true_labels[2] += label3.detach().cpu().numpy().tolist()

    test_weighted_f1 = utils.score_function(true_labels, model_preds)
    return np.mean(val_loss), test_weighted_f1

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    utils.seed_everything(options.SEED)
    if not os.path.isdir(options.MODEL_PATH):
        os.makedirs(options.MODEL_PATH)
    print('Train Step Start')
    isCleaned = True

    train_loader, val_loader, classes = createDataLoader('./data/train_cleaned.csv', isCleaned)
    print('Data Load Complete')
    kobert = AutoModel.from_pretrained("klue/roberta-large")
    print(kobert.embeddings.word_embeddings)
    model = TourismModel(kobert, 1024, classes)
    print('Model Load Complete')
    model.eval()

    infer_model = train(model, train_loader, val_loader, device)
