from typing import List

import numpy as np
import torch
from tqdm import tqdm

from callback.callback import Callback
from model.base_model import BaseModelApp


def epoch_train(
    model_app: BaseModelApp, train_loader, criterion, callbacks: List[Callback] = []
):
    model_app.train()
    losses = []
    for i, batch in (
        pbar := tqdm(enumerate(train_loader), total=len(train_loader), unit=" batch")
    ):
        batch_x = batch[0].to(torch.float32).to("cuda")
        batch_y = batch[1].to(torch.float32).to("cuda")
        model_app.zero_grad()
        pred_y = model_app.forward(batch_x, batch_y, is_training=True)

        loss = criterion(pred_y, batch_y)
        loss.backward()
        model_app.step()

        losses.append(loss.item())
        pbar.set_description(f"loss: {loss.item():.4f}")

        [cb.on_train_batch_end(pred_y, batch_y, loss.item()) for cb in callbacks]

    pbar.set_description(f"{np.mean(losses):.4f}")

    return (
        np.mean(losses),
        batch_y.cpu().detach().numpy(),
        pred_y.cpu().detach().numpy(),
    )


def epoch_val(
    model_app: BaseModelApp, val_loader, criterion, callbacks: List[Callback] = []
):
    model_app.eval()

    validation_losses = []
    batch_losses = []
    for i, (batch_x, batch_y) in enumerate(val_loader):
        batch_x = batch_x.to(torch.float32).to("cuda")
        batch_y = batch_y.to(torch.float32).to("cuda")
        pred_y = model_app.forward(batch_x, batch_y)
        loss = criterion(pred_y, batch_y)
        validation_losses.append(loss.item())
        batch_losses.append(loss.item())
        [cb.on_val_end(pred_y, batch_y, loss.item()) for cb in callbacks]

    return (
        np.mean(validation_losses),
        batch_y.cpu().detach().numpy(),
        pred_y.cpu().detach().numpy(),
    )


def predict(model_app: BaseModelApp, test_loader):
    model_app.eval()

    preds = []
    gts = []
    for i, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.to(torch.float32).to("cuda")
        batch_y = batch_y.to(torch.float32).to("cuda")
        pred_y = model_app.forward(batch_x, batch_y)

        preds.append(pred_y.cpu().detach().numpy())
        gts.append(batch_y.cpu().detach().numpy())

    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    gts = np.array(gts)
    gts = gts.reshape(-1, gts.shape[-2], gts.shape[-1])
    return preds, gts
