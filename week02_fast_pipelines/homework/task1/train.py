import torch
from torch import nn
from tqdm.auto import tqdm

from unet import Unet

from dataset import get_train_data

import argparse

class Scaler:
    def __init__(
        self,
        init_scale: float = 2**15,
        update_factor: float = 2.0,
        update_interval: int = 2000,
        dynamic_mode: bool = True
    ) -> None:
        """
        :param init_scale: initial loss scaling factor
        :param update_factor: increase factor for the scaling coefficient
        :param update_interval: number of consecutive successful steps before increasing the scaling factor
        :param dynamic_mode: whether to use dynamic or static loss scaling
        """
        self._scale = init_scale
        self._update_factor = update_factor
        self._update_interval = update_interval
        self._dynamic_mode = dynamic_mode

        self._found_nan_inf = False
        self._step = 0
    

    def scale(self, loss):
        """
        Scale the loss before backward pass.
        """
        return self._scale * loss
    

    @torch.no_grad()
    def _remove_scale(self, optimizer):
        """
        Unscale gradients.
        """
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad /= self._scale


    @torch.no_grad()
    def _check_nan_inf(self, optimizer):
        """
        Check whether any gradient contains NaN or Inf values.
        """
        self._found_nan_inf = False
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        self._found_nan_inf = True
                        return
    

    def step(self, optimizer):
        """
        Unscale gradients and perform an optimization step if gradients are valid.
        """
        self._check_nan_inf(optimizer)
        if self._found_nan_inf:
            optimizer.zero_grad(set_to_none=True)
        else:
            self._remove_scale(optimizer)
            optimizer.step()
    

    def update(self):
        """
        Update the loss scaling factor in dynamic mode.
        """
        if not self._dynamic_mode:
            return
        
        if self._found_nan_inf:
            self._scale /= self._update_factor
            self._step = 0
        else:
            self._step += 1
            if self._step % self._update_interval == 0:
                self._scale *= self._update_factor




def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    scaler: Scaler,
    device: torch.device,
) -> None:
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device.type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


def train(dynamic_mode: bool = True):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()

    scaler = Scaler(dynamic_mode=dynamic_mode)

    num_epochs = 5
    for _ in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, scaler, device=device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["static", "dynamic"], default="dynamic")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(dynamic_mode=(args.mode == "dynamic"))
