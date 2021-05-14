
import os
from copy import deepcopy

import numpy as np
import torch
from dival.reconstructors.standard_learned_reconstructor import (
    StandardLearnedReconstructor)
from dival.reconstructors.networks.unet import UNet
from odl.tomo import fbp_op

from .mask import Masker



# def toSubLists(full, ratio):
#     idx = set(
#         random.choices(
#             list(full),k=int(ratio*len(full))
#         )
#     )
#     idxC = list(full-idx)
#     idx = list(idx)
#     return idx, idxC

def toSubLists(full, ratio):
    a = list(full)
    return a[0:][::2], a[1:][::2]


class N2SelfReconstructor(StandardLearnedReconstructor):
    HYPER_PARAMS = deepcopy(StandardLearnedReconstructor.HYPER_PARAMS)
    HYPER_PARAMS.update({
        'scales': {
            'default': 5,
            'retrain': True
        },
        'skip_channels': {
            'default': 4,
            'retrain': True
        },
        'channels': {
            'default': (32, 32, 64, 64, 128, 128),
            'retrain': True
        },
        'filter_type': {
            'default': 'Hann',
            'retrain': True
        },
        'frequency_scaling': {
            'default': 1.0,
            'retrain': True
        },
        'use_sigmoid': {
            'default': False,
            'retrain': True
        },
        'init_bias_zero': {
            'default': True,
            'retrain': True
        },
        'lr': {
            'default': 0.001,
            'retrain': True
        },
        'scheduler': {
            'default': 'cosine',
            'choices': ['base', 'cosine'],  # 'base': inherit
            'retrain': True
        },
        'lr_min': {  # only used if 'cosine' scheduler is selected
            'default': 1e-4,
            'retrain': True
        }
    })
    DEVICE = 'cuda'
    DTYPE = torch.cuda.FloatTensor
    INPUT_DEPTH = 1
    IMAGE_DEPTH = 1
    IMAGE_SIZE = 512
    SHOW_EVERY = 50
    SAVE_EVERY = 1000

    def __init__(self, ray_trafo, **kwargs):
        super().__init__(ray_trafo, **kwargs)

    def train(self, dataset):
        super().train(dataset)
 
    def init_model(self):
        self.fbp_op = fbp_op(self.op, filter_type=self.filter_type,
                             frequency_scaling=self.frequency_scaling)
        self.model = UNet(in_ch=1, out_ch=1,
                          channels=self.channels[:self.scales],
                          skip_channels=[self.skip_channels] * (self.scales),
                          use_sigmoid=self.use_sigmoid)
        if self.init_bias_zero:
            def weights_init(m):
                if isinstance(m, torch.nn.Conv2d):
                    m.bias.data.fill_(0.0)
            self.model.apply(weights_init)
        
        if self.use_cuda:
            self.model = nn.DataParallel(self.model).to(self.device)

    def init_scheduler(self, dataset_train):
        if self.scheduler.lower() == 'cosine':
            # need to set private self._scheduler because self.scheduler
            # property accesses hyper parameter of same name,
            # i.e. self.hyper_params['scheduler']
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=self.lr_min)
        else:
            super().init_scheduler(dataset_train)   

    def _reconstruct(self, observation):
        self.model.eval()
        fbp = self.fbp_op(observation)
        fbp_tensor = torch.from_numpy(
            np.asarray(fbp)[None, None]).to(self.device)
        reco_tensor = self.model(fbp_tensor)
        reconstruction = reco_tensor.cpu().detach().numpy()[0, 0]
        return self.reco_space.element(reconstruction)

