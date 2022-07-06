import torch
import torch.nn as nn

from torch.nn.modules.loss import _Loss


class AdversarialLoss(_Loss):
    """
    GAN adversarial loss function.
    """
    def __init__(self, real_label_val: float = 1.0, fake_label_val: float = 0.0):
        super(AdversarialLoss, self).__init__()

        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss = nn.BCEWithLogitsLoss()

    def _get_target_label(self, input_t: torch.Tensor, target_is_real: bool):
        """
        Get target label.

        Args:
            input_t (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (Tensor): Target tensor.
        """
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input_t.new_ones(input_t.size()) * target_val

    def forward(self, input_t: torch.Tensor, target_is_real):
        """
        Forward method for the GAN adversarial loss.

        Args:
            input_t (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self._get_target_label(input_t, target_is_real)
        loss = self.loss(input_t, target_label)

        return loss
