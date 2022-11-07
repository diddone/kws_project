import torch
import numpy as np
from models import CRNN, CRNNPool
from base_config import BaseConfig
import dataclasses

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

# FA - true: 0, model: 1
# FR - true: 1, model: 0
def count_FA_FR(preds, labels):
    FA = torch.sum(preds[labels == 0])
    FR = torch.sum(labels[preds == 0])

    # torch.numel - returns total number of elements in tensor
    return FA.item() / torch.numel(preds), FR.item() / torch.numel(preds)

def get_au_fa_fr(probs, labels):
    sorted_probs, _ = torch.sort(probs)
    sorted_probs = torch.cat((torch.Tensor([0]), sorted_probs, torch.Tensor([1])))
    labels = torch.cat(labels, dim=0)

    FAs, FRs = [], []
    for prob in sorted_probs:
        preds = (probs >= prob) * 1
        FA, FR = count_FA_FR(preds, labels)
        FAs.append(FA)
        FRs.append(FR)
    # plt.plot(FAs, FRs)
    # plt.show()

    # ~ area under curve using trapezoidal rule
    return -np.trapz(FRs, x=FAs)


def save_model(model, path):
    torch.save({
        'model_state': model.state_dict(),
        'config_dict': dataclasses.asdict(model.config)
    }, path)

def load_model(path):
    load_dict = torch.load(path)
    config_dict = load_dict['config_dict']
    config = BaseConfig(**config_dict)
    model = CRNN(config)
    model.load_state_dict(load_dict['model_state'])

    return model

def load_model_pool(path):
    load_dict = torch.load(path)
    config_dict = load_dict['config_dict']
    config = BaseConfig(**config_dict)
    model = CRNNPool(config)
    model.load_state_dict(load_dict['model_state'])

    return model


def load_base_model():
    base_config = BaseConfig(hidden_size=32)

    base_model = CRNN(base_config)
    base_model.load_state_dict(torch.load('base_model.pth'))
    return base_model