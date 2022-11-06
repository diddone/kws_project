import torch
import torch.nn.functional as F
from torch import nn
from train import validation
from collections import defaultdict
from tqdm import tqdm
from utils import count_FA_FR
import wandb

# https://github.com/szagoruyko/attention-transfer/blob/master/utils.py#L10
def distillation_loss(student_scores, teacher_scores, labels, T, alpha):
    p = F.log_softmax(student_scores/T, dim=-1)
    q = F.softmax(teacher_scores/T, dim=-1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2)
    l_ce = F.cross_entropy(student_scores, labels)
    return l_kl * alpha + l_ce * (1. - alpha)

def train_epoch_distil(
    st_model, teacher_model, opt, loader, log_melspec,
    device, alpha=0.1, temp=5):
    st_model.train()
    teacher_model.eval()
    teacher_model.to(device)

    for i, (batch, labels) in tqdm(enumerate(loader), total=len(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = log_melspec(batch)

        opt.zero_grad()

        # run model # with autocast():
        st_logits = st_model(batch)

        with torch.no_grad():
            te_logits = teacher_model(batch)

        # we need probabilities so we use softmax & CE separately
        loss = distillation_loss(st_logits, te_logits, labels, T=temp, alpha=alpha)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(st_model.parameters(), 5)

        opt.step()

        # logging
        probs = F.softmax(st_logits, dim=-1)
        argmax_probs = torch.argmax(probs, dim=-1)
        FA, FR = count_FA_FR(argmax_probs, labels)
        acc = torch.sum(argmax_probs == labels) / torch.numel(argmax_probs)

    return acc

def train_distillation(
    st_model, te_model,
    train_loader, val_loader,
    melspec_train, melspec_val,
    opt, config, alpha, temp, scheduler=None):
    history = defaultdict(list)
    for n in range(config.num_epochs):

        train_acc = train_epoch_distil(st_model, te_model, opt, train_loader,
                    melspec_train, config.device, alpha=alpha, temp=temp)

        if scheduler is not None:
            scheduler.step()

        au_fa_fr = validation(st_model, val_loader,
                            melspec_val, config.device)
        history['val_metric'].append(au_fa_fr)

        if config.use_wandb:
            wandb_dict = {
                'train acc': train_acc,
                'valid metric': au_fa_fr,
                'epoch_number': n,
            }
            if scheduler is not None:
                wandb_dict['learning rate'] = scheduler.get_last_lr()[0]
            wandb.log(wandb_dict)
        # clear_output()
        # plt.plot(history['val_metric'])
        # plt.ylabel('Metric')
        # plt.xlabel('Epoch')
        # plt.grid()
        # plt.show()

        print('END OF EPOCH', n)
        print('Val metric', au_fa_fr)

    return history