import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable


class SmoothVanillaBackprop:
    def __init__(
        self,
        encoder,
        dim_l,
        source_id,
        saliency_n_iter=1,
        saliency_mean=0.,
        saliency_std=0.,
        saliency_use_relu=True,
        **kwargs
    ):
        super(SmoothVanillaBackprop, self).__init__()
        self._model = encoder
        self._n_iter = saliency_n_iter
        self._mean = saliency_mean
        self._std = saliency_std
        self._use_relu = saliency_use_relu
        self._dim_l = dim_l
        self._source_id = source_id
        self._model.eval()
        if torch.cuda.device_count() > 1:
            self._model = nn.DataParallel(self._model)
        self._model.cuda()
        self.hook_first_layer()
        self._gradients = {}

    def hook_first_layer(self):
        def hook_function(module, grad_in, grad_out):
            self._gradients[grad_in[0].device.index] = grad_in[0]


        if torch.cuda.device_count() > 1:
            first_layer = self._model.module._layers[0]._layer[0]
        else:
            first_layer = self._model._layers[0]._layer[0]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_dim):
        latent, _ = self._model(input_image)
        self._model.zero_grad()
        fake_latent = torch.FloatTensor(latent.size()).zero_().cuda()
        fake_latent[:, target_dim] = 1
        fake_latent = Variable(fake_latent)
        latent.backward(gradient=fake_latent)
        gradients_as_arr = None
        for key in sorted(list(self._gradients.keys())):
            if gradients_as_arr is None:
                gradients_as_arr = self._gradients[key].data.cpu()
            else:
                gradients_as_arr = torch.cat(
                    (
                        gradients_as_arr,
                        self._gradients[key].data.cpu()
                    ),
                    0
                )
        if self._use_relu:
            gradients_as_arr = torch.nn.functional.relu(gradients_as_arr)
            quantile = np.quantile(gradients_as_arr, 0.95)
            gradients_as_arr[gradients_as_arr < quantile] = 0

        return gradients_as_arr

    def forward(self, batch):
        def generate_noisy_batch(batch, mean, std):
            noise = torch.normal(size=batch.size(), mean=mean, std=std).cuda()
            noisy_batch = batch + noise
            return noisy_batch

        batch = batch[f'source_{self._source_id}']['data']
        smooth_grad = torch.zeros((batch.size(0), self._dim_l, *batch.size()[2:]))
        # setup the noise
        if self._std <= 0:
            logging.info('Using non noisy gradients. Set n_ter to 1.')
            self._n_iter = 1
        # calculate gradients
        for i in range(self._n_iter):
            logging.info(f'Performing gradient for {i}')
            if self._std > 0:
                noisy_batch = generate_noisy_batch(batch, self._mean, self._std)
            else:
                noisy_batch = batch
            noisy_batch = Variable(noisy_batch.data, requires_grad=True).cuda()
            for target_dim in tqdm(range(self._dim_l)):
                vanilla_grads = self.generate_gradients(
                    noisy_batch, target_dim)
                vanilla_grads = vanilla_grads.squeeze(1)
                smooth_grad[:, target_dim, :, :, :] = \
                    smooth_grad[:, target_dim, :, :, :] + vanilla_grads.cpu()
        # average the gradients
        smooth_grad = smooth_grad
        if self._n_iter > 1:
            smooth_grad = smooth_grad / self._n_iter
        return smooth_grad
