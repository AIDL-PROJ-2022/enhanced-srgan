import torch

from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class ContentLoss(_Loss):
    """
    Pixel-wise content loss function.

    Args:
       loss_f: Loss function to use to compute pixel-wise distance between images.

    Raises:
        NotImplementedError: `loss_f` must be one of: ``'l1'``, ``'l2'``, or ``'mse'``, raise error otherwise.
    """
    def __init__(self, loss_f: str = "l1"):
        super(ContentLoss, self).__init__()

        if loss_f.lower() in {"l1"}:
            self.loss_f = F.l1_loss
        elif loss_f.lower() in {"l2", "mse"}:
            self.loss_f = F.mse_loss
        else:
            raise NotImplementedError()

    def forward(self, out_images: torch.Tensor, gt_images: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation method for the content loss.

        Args:
            out_images: Batch of input (generated) images.
            gt_images: Batch of target (real, ground truth) images.

        Returns:
            Loss, scalar.
        """
        return self.loss_f(out_images, gt_images)
