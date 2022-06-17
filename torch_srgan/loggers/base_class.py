from torch import Tensor


class Logger:
    def log_stage(self, stage, epoch, train_losses, val_losses, psnr_metric, ssim_metric):
        raise NotImplementedError
    def log_generator_train_image(self, epoch:int, out_images: Tensor):
        raise NotImplementedError
