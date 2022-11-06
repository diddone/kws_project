import torch
import torch.nn.functional as F
from torch import nn
from utils import count_FA_FR, get_au_fa_fr
from tqdm import tqdm
from collections import defaultdict
import wandb

def train_epoch(model, opt, loader, log_melspec, device):
    model.train()
    for i, (batch, labels) in tqdm(enumerate(loader), total=len(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = log_melspec(batch)

        opt.zero_grad()

        # run model # with autocast():
        logits = model(batch)
        # we need probabilities so we use softmax & CE separately
        probs = F.softmax(logits, dim=-1)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        opt.step()

        # logging
        argmax_probs = torch.argmax(probs, dim=-1)
        FA, FR = count_FA_FR(argmax_probs, labels)
        acc = torch.sum(argmax_probs == labels) / torch.numel(argmax_probs)

    return acc

@torch.no_grad()
def validation(model, loader, log_melspec, device):
    model.eval()

    val_losses, accs, FAs, FRs = [], [], [], []
    all_probs, all_labels = [], []
    for i, (batch, labels) in tqdm(enumerate(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = log_melspec(batch)

        output = model(batch)
        # we need probabilities so we use softmax & CE separately
        probs = F.softmax(output, dim=-1)
        loss = F.cross_entropy(output, labels)

        # logging
        argmax_probs = torch.argmax(probs, dim=-1)
        all_probs.append(probs[:, 1].cpu())
        all_labels.append(labels.cpu())
        val_losses.append(loss.item())
        accs.append(
            torch.sum(argmax_probs == labels).item() /  # ???
            torch.numel(argmax_probs)
        )
        FA, FR = count_FA_FR(argmax_probs, labels)
        FAs.append(FA)
        FRs.append(FR)

    # area under FA/FR curve for whole loader
    au_fa_fr = get_au_fa_fr(torch.cat(all_probs, dim=0).cpu(), all_labels)
    return au_fa_fr


def train_base(
    model, train_loader, val_loader,
    melspec_train, melspec_val,
    opt, config, scheduler=None):
    history = defaultdict(list)
    for n in range(config.num_epochs):

        train_acc = train_epoch(model, opt, train_loader, melspec_train, config.device)

        if scheduler is not None:
            scheduler.step()

        au_fa_fr = validation(model, val_loader,
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
