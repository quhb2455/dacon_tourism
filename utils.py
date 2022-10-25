import random
import os
import numpy as np
import torch
from sklearn.metrics import f1_score
import glob
import options

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def score_function(real, pred):
    result = []
    for r, p in zip(real, pred):
        result.append(f1_score(r, p, average="weighted"))
    return result

def save_model(model, path, epoch):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict()},
               path)

def load_model(model, path, device):
    ckpt = torch.load(path)
    epoch = ckpt['epoch']
    model.load_state_dict(ckpt['model_state_dict'])#, strict=False)
    model.to(device)
    print(epoch, 'Model Load')
    return model, epoch

def latest_checkpoint_path(dir_path, regex="checkpoint_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x

def save_log(e, t_loss, v_loss, t_score, v_score):
    path = os.path.join(options.MODEL_PATH, 'train_log.txt')

    with open(path, 'a') as f:
        f.write(f'E {e} : train loss [{t_loss:.4f} val loss [{v_loss:.4f}] t_score [{t_score[0]:.2f},{t_score[1]:.2f},{t_score[2]:.2f}] v_score [{v_score[0]:.2f},{v_score[1]:.2f},{v_score[2]:.2f}]\n')
