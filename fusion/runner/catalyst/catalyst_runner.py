from catalyst import dl, metrics
from fusion.runner import ABaseRunner
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, \
    OneCycleLR, CyclicLR, ReduceLROnPlateau
import torch
import torch.nn.functional as F
from typing import Mapping, Any


class CatalystRunner(ABaseRunner, dl.Runner):
    epoch = 0
    train_batch_id = 0
    valid_batch_id = 0
    valid_loss = torch.zeros(1)

    def predict_batch(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """

        Args:
            batch:
            kwargs:

        Returns:

        """
        x, y = self._unpack_batch(batch)
        return self.model(x), y

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        """
        Args:
            batch:
            
        Returns:
        """
        if self.is_train_loader:
            for param in self.model.parameters():
                param.grad = None

        x, y = self._unpack_batch(batch)
        outputs = self.model(x)
        if self.criterion:
            loss = self.criterion(outputs, y)

            if isinstance(loss, tuple):
                loss, raw_losses = loss
                self.batch_metrics.update(raw_losses)

            if self.is_train_loader:
                self.train_batch_id += 1
                loss.backward()
                self.optimizer.step()
                if isinstance(self.scheduler, (CosineAnnealingWarmRestarts)):
                    self.scheduler.step(
                        epoch=int(self.epoch + self.train_batch_id / len(self._loaders['train'])))
                elif isinstance(self.scheduler, (OneCycleLR, CyclicLR)):
                    self.scheduler.step()
                else:
                    if self.scheduler is not None and not isinstance(self.scheduler, (ReduceLROnPlateau)):
                        raise NotImplementedError
            else:
                if self.loader_key == 'valid':
                    self.valid_batch_id += 1
                    self.valid_loss += loss.item()
                    if (self.valid_batch_id // len(self._loaders['valid'])) == 1:
                        if isinstance(self.scheduler, (ReduceLROnPlateau)):
                            mean_val_loss = self.valid_loss / len(self._loaders['valid'])
                            self.scheduler.step(mean_val_loss)

            self.batch_metrics["loss"] = loss.item()
            for key in ["loss"]:
                self.meters[key].update(self.batch_metrics[key], self.batch_size)

        self.batch = {"targets": y}
        for source_id, source_z in outputs.z.items():
            probs = F.softmax(source_z, dim=1)
            self.batch[f"logits_{source_id}"] = source_z
            self.batch[f"probs_{source_id}"] = probs

    def get_loaders(self):
        return self._loaders

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False) for key in ["loss"]
        }

    def on_loader_end(self, runner):
        for key in ["loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        if self.is_train_loader:
            self.epoch += 1
            self.train_batch_id = 0
        else:
            if self.loader_key == 'valid':
                self.valid_loss = torch.zeros(1)
                self.valid_batch_id = 0
        super().on_loader_end(runner)

    def _unpack_batch(self, batch):
        x, y = batch
        return x, y
