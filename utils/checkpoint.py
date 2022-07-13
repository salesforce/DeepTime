from typing import Optional, Dict
import logging
from os.path import join

import gin
import torch
from torch.utils.tensorboard import SummaryWriter


@gin.configurable()
class Checkpoint:
    def __init__(self,
                 checkpoint_dir: str,
                 patience: Optional[int] = 7,
                 delta: Optional[float] = 0.):
        self.checkpoint_dir = checkpoint_dir
        self.model_path = join(checkpoint_dir, 'model.pth')

        # early stopping
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.delta = delta

        # logging
        self.summary_writer = SummaryWriter(log_dir=checkpoint_dir)

    def __call__(self,
                 epoch: int,
                 model: torch.nn.Module,
                 scalars: Optional[Dict[str, float]] = None):
        for name, value in scalars.items():
            # logging
            self.summary_writer.add_scalar(name, value, epoch)

            # early stopping
            if name == 'Loss/Val':
                val_loss = value
                if val_loss <= self.best_loss + self.delta:
                    logging.info(
                        f"Validation loss decreased ({self.best_loss:.3f} --> {val_loss:.3f}). Saving model ...")
                    torch.save(model.state_dict(), self.model_path)
                    self.best_loss = val_loss
                    self.counter = 0
                else:
                    self.counter += 1
                    logging.info(f"Validation loss increased ({self.best_loss:.3f} --> {val_loss:.3f}). "
                                 f"Early stopping counter: {self.counter} out of {self.patience}")
                    if self.counter >= self.patience >= 0:
                        self.early_stop = True

        self.summary_writer.flush()

    def close(self, scores: Optional[Dict[str, float]] = None):
        if scores is not None:
            for name, value in scores.items():
                self.summary_writer.add_scalar(name, value)
        self.summary_writer.close()