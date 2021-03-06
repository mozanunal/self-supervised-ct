
import os
from copy import deepcopy
from math import ceil

import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    TENSORBOARD_AVAILABLE = False
else:
    TENSORBOARD_AVAILABLE = True
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR
from odl.tomo import fbp_op
from odl.contrib.torch import OperatorModule
from dival.reconstructors.standard_learned_reconstructor import (
    StandardLearnedReconstructor)
from dival.reconstructors.networks.unet import UNet
# from .unet import UNet
from dival.measure import PSNR, SSIM

from .mask import Masker
from .tool import np_to_torch, torch_to_np



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

    def __init__(self, ray_trafo, **kwargs):
        super().__init__(ray_trafo, **kwargs)
        self.ray_trafo = ray_trafo
        self.ray_trafo_module = OperatorModule(self.ray_trafo)
        self.init_model()

    def _train_one_epoch(self):
        pass

    def train(self, dataset):
        if self.torch_manual_seed:
            torch.random.manual_seed(self.torch_manual_seed)

        self.init_transform(dataset=dataset)

        # create PyTorch datasets
        dataset_train = dataset.create_torch_dataset(
            part='train', reshape=((1,) + dataset.space[0].shape,
                                   (1,) + dataset.space[1].shape),
            transform=self._transform)

        dataset_validation = dataset.create_torch_dataset(
            part='validation', reshape=((1,) + dataset.space[0].shape,
                                        (1,) + dataset.space[1].shape))

        # reset model before training
        

        criterion = torch.nn.MSELoss() #torch.nn.MSELoss()
        self.init_optimizer(dataset_train=dataset_train)

        # create PyTorch dataloaders
        shuffle = (dataset.supports_random_access() if self.shuffle == 'auto'
                   else self.shuffle)
        data_loaders = {
            'train': DataLoader(
                dataset_train, batch_size=self.batch_size,
                num_workers=self.num_data_loader_workers, shuffle=shuffle,
                pin_memory=True, worker_init_fn=self.worker_init_fn),
            'validation': DataLoader(
                dataset_validation, batch_size=self.batch_size,
                num_workers=self.num_data_loader_workers, shuffle=shuffle,
                pin_memory=True, worker_init_fn=self.worker_init_fn)}

        dataset_sizes = {'train': len(dataset_train),
                         'validation': len(dataset_validation)}

        self.init_scheduler(dataset_train=dataset_train)
        if self._scheduler is not None:
            schedule_every_batch = isinstance(
                self._scheduler, (CyclicLR, OneCycleLR))

        best_model_wts = deepcopy(self.model.state_dict())
        best_psnr = 0

        if self.log_dir is not None:
            if not TENSORBOARD_AVAILABLE:
                raise ImportError(
                    'Missing tensorboard. Please install it or disable '
                    'logging by specifying `log_dir=None`.')
            writer = SummaryWriter(log_dir=self.log_dir, max_queue=0)
            validation_samples = dataset.get_data_pairs(
                'validation', self.log_num_validation_samples)

        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_psnr = 0.0
                running_ssim = 0.0
                running_loss = 0.0
                running_size = 0
                num_iter = 0
                with tqdm(data_loaders[phase],
                          desc='epoch {:d}'.format(epoch + 1),
                          disable=not self.show_pbar) as pbar:
                    for inputs, labels in pbar:
                        num_iter += 1
                        if self.normalize_by_opnorm:
                            inputs = (1./self.opnorm) * inputs
                        net_input, mask = self.masker.mask( inputs, num_iter % (self.masker.n_masks - 1) )
                        #loss_ratio = torch.numel(mask) / mask.sum()
                        #mask *= loss_ratio
                        # fbp reconstruct
                        initial_outputs = np.zeros(labels.shape, dtype=np.float32)
                        for i in range(len(inputs)):
                            initial_outputs[i,0,:,:] = self.fbp_op(net_input[i,0].numpy())
                        initial_outputs = torch.from_numpy(initial_outputs)


                        initial_outputs = initial_outputs.to(self.device)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        mask = mask.to(self.device)

                        # zero the parameter gradients
                        self._optimizer.zero_grad()

                        # forward
                        # track gradients only if in train phase
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(initial_outputs)
                            loss = criterion( 
                                self.ray_trafo_module(outputs)*mask,
                                inputs*mask)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm=1)
                                self._optimizer.step()
                                if (self._scheduler is not None and
                                        schedule_every_batch):
                                    self._scheduler.step()

                        for i in range(outputs.shape[0]):
                            labels_ = labels[i, 0].detach().cpu().numpy()
                            outputs_ = outputs[i, 0].detach().cpu().numpy()
                            running_psnr += PSNR(outputs_, labels_)
                            running_ssim += SSIM(outputs_, labels_)

                        # statistics
                        running_loss += loss.item() * outputs.shape[0]
                        running_size += outputs.shape[0]

                        pbar.set_postfix({'phase': phase,
                                          'loss': running_loss/running_size,
                                          'psnr': running_psnr/running_size,
                                          'ssim': running_ssim/running_size})
                        if self.log_dir is not None and phase == 'train':
                            step = (epoch * ceil(dataset_sizes['train']
                                                 / self.batch_size)
                                    + ceil(running_size / self.batch_size))
                            writer.add_scalar(
                                'loss/{}'.format(phase),
                                torch.tensor(running_loss/running_size), step)
                            writer.add_scalar(
                                'psnr/{}'.format(phase),
                                torch.tensor(running_psnr/running_size), step)
                            writer.add_scalar(
                                'ssim/{}'.format(phase),
                                torch.tensor(running_ssim/running_size), step) 

                    if (self._scheduler is not None
                            and not schedule_every_batch):
                        self._scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_psnr = running_psnr / dataset_sizes[phase]
                    epoch_ssim = running_ssim / dataset_sizes[phase]

                    if self.log_dir is not None and phase == 'validation':
                        step = (epoch+1) * ceil(dataset_sizes['train']
                                                / self.batch_size)
                        writer.add_scalar('loss/{}'.format(phase),
                                          epoch_loss, step)
                        writer.add_scalar('psnr/{}'.format(phase),
                                          epoch_psnr, step)
                        writer.add_scalar('ssim/{}'.format(phase),
                                          epoch_ssim, step)
                    # deep copy the model (if it is the best one seen so far)
                    if phase == 'validation' and epoch_psnr > best_psnr:
                        best_psnr = epoch_psnr
                        best_model_wts = deepcopy(self.model.state_dict())
                        if self.save_best_learned_params_path is not None:
                            self.save_learned_params(
                                self.save_best_learned_params_path)

                if (phase == 'validation' and self.log_dir is not None and
                        self.log_num_validation_samples > 0):
                    with torch.no_grad():
                        val_images = []
                        for (y, x) in validation_samples:
                            y = torch.from_numpy(
                                np.asarray(y))[None, None].to(self.device)
                            x = torch.from_numpy(
                                np.asarray(x))[None, None].to(self.device)
                            reco = self.model(y)
                            reco -= torch.min(reco)
                            reco /= torch.max(reco)
                            val_images += [reco, x]
                        writer.add_images(
                            'validation_samples', torch.cat(val_images),
                            (epoch + 1) * (ceil(dataset_sizes['train'] /
                                                self.batch_size)),
                            dataformats='NCWH')

        print('Best val psnr: {:4f}'.format(best_psnr))
        self.model.load_state_dict(best_model_wts)
 
    def init_model(self):
        self.fbp_op = fbp_op(self.op, filter_type=self.filter_type,
                             frequency_scaling=self.frequency_scaling)
        self.model = UNet(num_input_channels=1, num_output_channels=1,
                    feature_scale=4, more_layers=0, concat_x=False,
                    upsample_mode='bilinear', norm_layer=torch.nn.BatchNorm2d,
                    pad='reflect',
                    need_sigmoid=False, need_bias=True).to('cuda')
                    # UNet(in_ch=1, out_ch=1,
                    #       channels=self.channels[:self.scales],
                    #       skip_channels=[self.skip_channels] * (self.scales),
                    #       use_sigmoid=self.use_sigmoid)
        self.masker = Masker(width = 4, mode='interpolate')

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

