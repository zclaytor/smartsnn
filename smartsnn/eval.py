import os
import pkg_resources

import numpy as np
import torch

from .model import ConvNet


def load_model(model, device="cpu"):
    run_number = int(model[-4])
    path = os.path.join("models", model)
    stream = pkg_resources.resource_stream(__name__, path)

    if run_number == 0:
        c = [8, 16, 32]
    elif run_number == 1:
        c = [16, 32, 64]
    elif run_number == 2:
        c = [32, 64, 128]
    elif run_number == 3:
        c = [64, 128, 256]

    model = ConvNet(c)
    device = torch.device(device)
    model.load_state_dict(torch.load(stream, map_location=device))
    model.eval()
    return model

def evaluate(data, modelpath):
    pmax = int(modelpath[-8:-5])
    model = load_model(modelpath)
    
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    if data.dtype is not torch.float32:
        data = data.to(torch.float32)
    if len(data.shape) != 4:
        data = data.unsqueeze(1)
    if data.max() > 1:
        data /= data.max()

    preds = pmax * model(data).detach().numpy()
    return preds

def quick_eval(files, modelpath):
    from pandas import DataFrame

    tics = [int(x[x.find("TIC")+3:x.find("TIC")+14]) for x in files]

    y = torch.tensor(np.array([np.load(f) for f in files])/255).to(torch.float32).unsqueeze(1)
    preds = DataFrame(evaluate(y, modelpath), columns=["prot", "e_prot"])
    preds.index = tics
    preds.index.name = "TIC"
    
    return preds