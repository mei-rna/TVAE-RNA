# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""
import os
import torch
from .baseline import Baseline
from ..pretrained import load_hub_workaround

def build_model(cfg):
    model = Baseline(
        backbone_name=cfg.MODEL.BACKBONE_NAME,
        pairwise_predictor_name=cfg.MODEL.PAIRWISE_PREDICTOR_NAME,
        backbone_frozen=cfg.MODEL.BACKBONE_FROZEN,
        model_location=cfg.MODEL.MODEL_LOCATION,
    )
    return model



def build_rnafm_resnet(type="ss", model_location=None, fm_model_location=None, on_cpu=False):
    """
    :param type: for specific task type, like secondary structure prediction
    :return:
    """
    if type == "ss":
        model = Baseline(
            backbone_name="rna-fm",
            pairwise_predictor_name="pc-resnet_1_sym_first:r-ss",
            backbone_frozen=1,
            model_location=fm_model_location,
        )
        url = f"https://proj.cse.cuhk.edu.hk/rnafm/api/download?filename=RNA-FM-ResNet_PDB-All.pth"
    else:
        raise Exception("Unknown Model Type!")
    
    if model_location is None:
        model_state_dict = load_hub_workaround(url, download_name="RNA-FM-ResNet_PDB-All.pth")
        model.load_state_dict(model_state_dict)
    elif model_location is not None and os.path.exists(model_location):
        if on_cpu:
            model_state_dict = torch.load(model_location, map_location=torch.device('cpu'))
        else:
            model_state_dict = torch.load(model_location)
        model.load_state_dict(model_state_dict)
    else:
        raise Exception("Wrong Local Location of Model Given")

    return model, model.backbone_alphabet




