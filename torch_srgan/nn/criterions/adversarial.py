import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


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

    def __init__(self, real_label_val: float, fake_label_val: float):
        super(RelativisticAdversarialLoss, self).__init__()

        self.real_label_val = float(real_label_val)
        self.fake_label_val = float(fake_label_val)

    def forward(self, fake_logits: torch.Tensor, real_logits: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation method for the relativistic adversarial loss.

        Args:
            fake_logits: Probability that generated samples are not real.
            real_logits: Probability that real (ground truth) samples are fake.

        Returns:
            Loss, scalar.
        """
        # Compute relativistic adversarial loss for fake and real images
        loss_real_input = (real_logits - fake_logits.mean(0, keepdim=True))
        loss_real = F.binary_cross_entropy_with_logits(
            input=loss_real_input, target=torch.full_like(real_logits, self.real_label_val)
        )
        loss_fake_input = (fake_logits - real_logits.mean(0, keepdim=True))
        loss_fake = F.binary_cross_entropy_with_logits(
            input=loss_fake_input, target=torch.full_like(fake_logits, self.fake_label_val)
        )

        # Return the average value between this two losses
        return (loss_real + loss_fake) / 2
