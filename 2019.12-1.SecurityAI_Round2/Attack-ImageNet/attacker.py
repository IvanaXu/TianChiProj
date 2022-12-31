from typing import Optional, Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Attacker:
    def __init__(self,
                 steps: int,
                 quantize: bool = True,
                 levels: int = 256,
                 max_norm: Optional[float] = None,
                 div_prob: float = 0.9,
                 loss_amp: float = 4.0,
                 device: torch.device = torch.device('cpu')) -> None:
        self.steps = steps

        self.quantize = quantize
        self.levels = levels
        self.max_norm = max_norm
        self.div_prob = div_prob
        self.loss_amp = loss_amp
        
        self.device = device

    def input_diversity(self, image, low=270, high=299):
        if random.random() > self.div_prob:
            return image
        rnd = random.randint(low, high)
        rescaled = F.interpolate(image, size=[rnd, rnd], mode='bilinear')
        h_rem = high - rnd
        w_rem = high - rnd
        pad_top = random.randint(0, h_rem)
        pad_bottom = h_rem - pad_top
        pad_left = random.randint(0, w_rem)
        pad_right = w_rem - pad_left
        padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)
        return padded

    def attack(self, 
               model: nn.Module, 
               inputs: torch.Tensor, 
               labels_true: torch.Tensor,
               labels_target: torch.Tensor)-> torch.Tensor:

        batch_size = inputs.shape[0]
        delta = torch.zeros_like(inputs, requires_grad=True)

        # setup optimizer
        optimizer = optim.SGD([delta], lr=1, momentum=0.9)

        # for choosing best results
        best_loss = 1e4 * torch.ones(inputs.size(0), dtype=torch.float, device=self.device)
        best_delta = torch.zeros_like(inputs)

        for _ in range(self.steps):
            if self.max_norm:
                delta.data.clamp_(-self.max_norm, self.max_norm)
                if self.quantize:
                    delta.data.mul_(self.levels - 1).round_().div_(self.levels - 1)

            adv = inputs + delta
            div_adv = self.input_diversity(adv)

            logits = model(div_adv)

            ce_loss_true = F.cross_entropy(logits, labels_true, reduction='none')     
            ce_loss_target = F.cross_entropy(logits, labels_target, reduction='none')

            # fuse targeted and untargeted
            loss = self.loss_amp * ce_loss_target - ce_loss_true
            
            is_better = loss < best_loss

            best_loss[is_better] = loss[is_better]
            best_delta[is_better] = delta.data[is_better]
            
            loss = torch.mean(loss)
            optimizer.zero_grad()
            loss.backward()

            # renorm gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=float('inf'), dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            optimizer.step()

            # avoid out of bound
            delta.data.add_(inputs)
            delta.data.clamp_(0, 1).sub_(inputs)

        return inputs + best_delta
