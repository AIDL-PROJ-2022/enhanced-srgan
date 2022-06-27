import torch
import torch.nn as nn

from torch.nn import functional as F
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


class RelativisticAdversarialLoss(_Loss):
    """
    Relativistic average GAN loss function.

    It has been proposed in `The relativistic discriminator: a key element
    missing from standard GAN`_.

    Args:
        real_label_val: The value for real images label.
        fake_label_val: The value for fake images label.

    .. _`The relativistic discriminator: a key element missing
        from standard GAN`: https://arxiv.org/pdf/1807.00734.pdf
    """

    def __init__(self, label_noise_range: float):
        super(RelativisticAdversarialLoss, self).__init__()

        # self.real_label_val = float(real_label_val)
        # self.fake_label_val = float(fake_label_val)
        self.label_noise_range = float(label_noise_range)
        # self.is_discriminator = is_discriminator

    # def forward(self, fake_logits: torch.Tensor, real_logits: torch.Tensor) -> torch.Tensor:
    #     """
    #     Forward propagation method for the relativistic adversarial loss.
    #
    #     Args:
    #         fake_logits: Probability that generated samples are not real.
    #         real_logits: Probability that real (ground truth) samples are fake.
    #
    #     Returns:
    #         Loss, scalar.
    #     """
    #     # Generate label tensors with nosing (center the noise between +/- label_noise_range)
    #     real_labels_noise = ((torch.rand_like(real_logits) - 0.5) * 2.0) * self.label_noise_range
    #     real_labels = torch.full_like(real_logits, 1.) + real_labels_noise
    #     # Generate label tensors with nosing (add a maximum of label_noise_range to fake label)
    #     fake_labels_noise = torch.rand_like(fake_logits) * self.label_noise_range
    #     fake_labels = torch.full_like(fake_logits, 0.) + fake_labels_noise
    #     # Compute relativistic adversarial loss for fake and real images
    #     loss_real_input = (real_logits - fake_logits.mean(0, keepdim=True))
    #     loss_real = F.binary_cross_entropy_with_logits(
    #         input=loss_real_input, target=real_labels
    #     )
    #     loss_fake_input = (fake_logits - real_logits.mean(0, keepdim=True))
    #     loss_fake = F.binary_cross_entropy_with_logits(
    #         input=loss_fake_input, target=fake_labels
    #     )
    #
    #     # Return the average value between this two losses
    #     return (abs(loss_real) + abs(loss_fake)) / 2

    def _real_logits_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor):
        # Generate label tensors with nosing (center the noise between +/- label_noise_range)
        real_labels_noise = torch.rand_like(real_logits) * self.label_noise_range
        real_labels = torch.ones_like(real_logits) - real_labels_noise
        # real_labels = torch.ones_like(real_logits)

        # Compute relativistic adversarial loss for real images
        loss = F.binary_cross_entropy_with_logits(
            input=(real_logits - fake_logits.mean(0, keepdim=True)), target=real_labels
        )

        return loss

    def _fake_logits_loss(self, fake_logits: torch.Tensor, real_logits: torch.Tensor):
        # Generate label tensors with nosing (add a maximum of label_noise_range to fake label)
        fake_labels_noise = torch.rand_like(fake_logits) * self.label_noise_range
        fake_labels = torch.zeros_like(fake_logits) + fake_labels_noise
        # fake_labels = torch.zeros_like(fake_logits)

        # Compute relativistic adversarial loss for real images
        loss = F.binary_cross_entropy_with_logits(
            input=(fake_logits - real_logits.mean(0, keepdim=True)), target=fake_labels
        )

        return loss

    def forward(self, fake_logits: torch.Tensor, real_logits: torch.Tensor) -> torch.Tensor:
        real_logits_loss = self._real_logits_loss(fake_logits, real_logits)
        fake_logits_loss = self._fake_logits_loss(fake_logits, real_logits)

        return real_logits_loss + fake_logits_loss
