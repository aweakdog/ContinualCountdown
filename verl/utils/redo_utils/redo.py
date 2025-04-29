import torch
import torch.nn as nn
import math

class VerlGradientReDo:
    """
    verl-compatible Gradient-based ReDo implementation (neuron reset based on gradient statistics).
    """
    def __init__(self, model, tau=0.0, use_lecun_init=False, frequency=1000, optimizer=None, reset_steps=None):
        self.model = model
        self.tau = tau
        self.use_lecun_init = use_lecun_init
        self.frequency = frequency
        self.optimizer = optimizer
        self.reset_steps = reset_steps
        self.current_step = 0

    def _kaiming_uniform_reinit(self, layer, mask):
        fan_in = nn.init._calculate_correct_fan(layer.weight, mode="fan_in")
        gain = nn.init.calculate_gain("relu", math.sqrt(5))
        std = gain / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            layer.weight.data[mask, ...] = torch.empty_like(layer.weight.data[mask, ...]).uniform_(-bound, bound)
            if layer.bias is not None:
                layer.bias.data[mask] = 0

    def _get_redo_masks(self, tau):
        # Example: mask neurons whose grad norm < tau
        masks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                grad = module.weight.grad
                if grad is not None:
                    grad_norm = grad.norm(dim=1)
                    mask = grad_norm < tau
                    masks.append((module, mask))
        return masks

    def step(self):
        self.current_step += 1
        if self.current_step % self.frequency != 0:
            return
        redo_masks = self._get_redo_masks(self.tau)
        for module, mask in redo_masks:
            self._kaiming_uniform_reinit(module, mask)
