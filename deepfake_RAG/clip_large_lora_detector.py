'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the CLIPDetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International conference on machine learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}
'''

import os
import os
import math
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

import loralib as lora
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig

logger = logging.getLogger(__name__)


# @DETECTOR.register_module(module_name='clip_large_lora')
class CLIP_Large_LoRA_Detector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = nn.Linear(1024, 2)
        self.loss_func = nn.CrossEntropyLoss()
        
    def build_backbone(self, config):
        # prepare the backbone
        _, backbone = get_clip_visual(model_name="openai/clip-vit-large-patch14")
        backbone = to_lora(backbone, r=16)
        return backbone
        
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        feat = self.backbone(data_dict['image'])['pooler_output']
        return feat

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        
        # compute metrics for batch data
        # auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        # metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        
        acc, mAP = calculate_acc_for_train(label.detach(), pred.detach(), self.config['backbone_config']['num_classes'])
        metric_batch_dict = {'acc': acc, 'mAP': mAP}
        
        return metric_batch_dict
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        # prob = torch.softmax(pred, dim=1)[:, 1]
        prob = torch.softmax(pred, dim=1)
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict


def get_clip_visual(model_name = "openai/clip-vit-large-patch14"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return processor, model.vision_model


def to_lora(model, target=[nn.Linear, nn.Conv2d, nn.Embedding], r=16, f_class=None, layers=None, names=None):

    for n, m in model.named_modules():
        if f_class is not None and not isinstance(m, f_class):
            continue
        if isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList):
            
            for name, mod in m.named_children():
                
                # print(name, mod)
                if isinstance(mod, nn.Linear) and not isinstance(mod, lora.Linear):
                    mod = change_mod(mod, r=r)
                    m._modules[name] = mod
        else:
            if layers is None or any(['layers.' + str(i) in n for i in layers]):
                for name, mod in m.named_children():
                    # if 'self_attn' in f_name:
                    #     print(name, mod)
                    if isinstance(mod, nn.Linear) and not isinstance(mod, lora.Linear):
                        if names is None or any(na in name for na in names):
                            mod = change_mod(mod, r=r)
                            setattr(m, name, mod)
    
    lora.mark_only_lora_as_trainable(model)
    return model


def change_mod(m, targets=[nn.Linear, nn.Conv2d, nn.Embedding], r=16):
    st_dict = m.state_dict()
    
    if nn.Linear in targets and isinstance(m, nn.Linear):
        dtype = m.weight.dtype
        new_m = lora.Linear(m.in_features, m.out_features, bias=m.bias is not None, r=r, dtype=dtype)
        new_m.load_state_dict(st_dict, strict=False)
        # print(new_m)
        m = new_m
    elif nn.Conv2d in targets and isinstance(m, nn.Conv2d):
        new_m = lora.Conv2d(m.in_channels, m.out_channels, m.kernel_size, stride=m.stride, padding=m.padding, \
                        dilation=m.dilation, transposed=m.transposed, output_padding=m.output_padding, groups=m.groups, bias=m.bias, r=r)
        new_m.load_state_dict(st_dict, strict=False)
        m = new_m
    elif nn.Embedding in targets and isinstance(m, nn.Embedding):
        new_m = lora.Embedding(m.num_embeddings, m.embedding_dim, padding_idx=m.padding_idx, max_norm=m.max_norm, norm_type=m.norm_type, \
                scale_grad_by_freq=m.scale_grad_by_freq, freeze=m.freeze, sparse=m.sparse, r=r)
        new_m.load_state_dict(st_dict, strict=False)
        m = new_m

    return m
