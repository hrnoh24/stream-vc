from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    # spectral_reconstruction_loss,
    ReconstructionLoss
)


class StreamVCModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        scheduler_g: torch.optim.lr_scheduler,
        scheduler_d: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param generator: The generator model to train.
        :param discriminator: The discriminator model to train.
        :param optim_g: The optimizer to use for training the generator.
        :param optim_d: The optimizer to use for training the discriminator.
        :param scheduler_g: The learning rate scheduler to use for training the generator.
        :param scheduler_d: The learning rate scheduler to use for training the discriminator.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['generator', 'discriminator'])
        self.automatic_optimization = False

        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d

        self.criterion = torch.nn.CrossEntropyLoss()
        self.reconstruction_loss = ReconstructionLoss()

    def forward(
        self, x: torch.Tensor, pitch: torch.Tensor, energy: torch.Tensor
    ) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.generator(x, pitch=pitch, energy=energy, train=True)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # self.val_loss.reset()
        # self.val_acc.reset()
        # self.val_acc_best.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        y, pitch, energy, labels = batch

        optimizer_g, optimizer_d = self.optimizers()

        # Train discriminator
        self.toggle_optimizer(optimizer_d)
        y_hat, logits = self.forward(y, pitch, energy)
        y_d_hat_rs, y_d_hat_gs, _, _ = self.discriminator(y, y_hat.detach())
        loss_disc, _, _ = discriminator_loss(y_d_hat_rs, y_d_hat_gs)

        self.log("train/d_loss", loss_disc, prog_bar=True)
        self.manual_backward(loss_disc)
        self.clip_gradients(optimizer_d, gradient_clip_val=1, gradient_clip_algorithm="norm")
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        # Train generator
        self.toggle_optimizer(optimizer_g)
        y_hat, logits = self.forward(y, pitch, energy)
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(y, y_hat)

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_hat_g)
        loss_recon = self.reconstruction_loss(y, y_hat)
        loss_content = self.criterion(logits.transpose(1, 2), labels)
        loss_all = 100 * loss_fm + loss_gen + loss_recon + loss_content
        self.manual_backward(loss_all)
        self.clip_gradients(optimizer_g, gradient_clip_val=1, gradient_clip_algorithm="norm")
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        self.log_dict(
            {
                "train/g_loss": loss_gen,
                "train/fm_loss": loss_fm,
                "train/recon_loss": loss_recon,
                "train/content_loss": loss_content,
                "train/loss": loss_all,
            },
            prog_bar=True,
        )

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        y, pitch, energy, labels = batch

        y_hat, logits = self.forward(y, pitch, energy)
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(y, y_hat)

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_hat_g)
        loss_recon = self.reconstruction_loss(y, y_hat)
        loss_content = self.criterion(logits.transpose(1, 2), labels)
        loss_all = 100 * loss_fm + loss_gen + loss_recon + loss_content

        self.log_dict(
            {
                "val/g_loss": loss_gen,
                "val/fm_loss": loss_fm,
                "val/recon_loss": loss_recon,
                "val/content_loss": loss_content,
                "val/loss": loss_all,
            },
            prog_bar=True,
        )

        # update and log metrics
        # self.val_loss(loss)
        # self.val_acc(preds, targets)
        # self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    # def setup(self, stage: str) -> None:
    #     """Lightning hook that is called at the beginning of fit (train + validate), validate,
    #     test, or predict.

    #     This is a good hook when you need to build models dynamically or adjust something about
    #     them. This hook is called on every process when using DDP.

    #     :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
    #     """
    #     if self.hparams.compile and stage == "fit":
    #         self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer_g = self.optimizer_g(params=self.generator.parameters())
        if self.scheduler_g is not None:
            scheduler_g = self.scheduler_g(optimizer=optimizer_g)

        optimizer_d = self.optimizer_d(params=self.discriminator.parameters())
        if self.scheduler_d is not None:
            scheduler_d = self.scheduler_d(optimizer=optimizer_d)
        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]


if __name__ == "__main__":
    pass
