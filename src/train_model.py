import os
import random
from datetime import datetime

import torch
from torch import nn
from tqdm import tqdm

from .utils import evaluate, imsave, LossTracker


def TVRegularization(scaling=1, in_channels=1):
    """Total variation regularization implementation using torcn.nn.Conv2d() layer"""

    sobel = nn.Conv2d(in_channels, 2, (3,3), padding=1, bias=False)
    sobel.weight.requires_grad_(False)
    sobel_x = torch.Tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]]).expand((in_channels, -1, -1))
    # Standardize filter, '4' acounts for square sum of gradients later
    sobel_x.div_(sobel_x.std() *4)
    sobel.weight.data = torch.stack([sobel_x, torch.transpose(sobel_x, -2,-1)], 
                                        axis=0).cuda().view(2, in_channels, 3, 3)

    def compute_TV(image):
        return torch.norm(torch.sqrt(sobel(image).pow(2).sum(dim=1)), 1) /image.numel() *scaling

    return compute_TV


def train(model, dataloaders, loss_criterion, epochs, regularization=None, **kwargs):

    def save_val_sample_pred(dataloader, n_samples=10):
        sample_ids = random.sample(range(len(dataloader.dataset)), n_samples)

        def pred_samples(model, filename):           
            sample_ims = (dataloader.dataset[i] for i in sample_ids)

            with evaluate(model):
                preds = torch.cat([torch.cat([sample, model(sample.unsqueeze(0)).squeeze(0), truth], dim=-2)
                                   for sample, truth in sample_ims],
                                  dim=-1)

            imsave(filename, preds.cpu().squeeze().numpy())

        return pred_samples


    save_folder = kwargs.get(
        'save_folder', f"model_weights/{model.__class__.__name__}_{datetime.now().strftime('%m%d%H%M%S')}")
    os.makedirs(save_folder)
    print(f"Saving to {save_folder}", flush=True)

    train_dl, val_dl = dataloaders

    optimizer = torch.optim.Adam(model.msd.parameters(), lr=kwargs.get('lr', 1e-3))
    gamma = 1e-2 **(1/(epochs-10))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Evaluate starting state of model
    compute_loss = lambda t: loss_criterion(model(t[0]), t[1]).item()
    best_val = sum(map(compute_loss, val_dl)) /len(val_dl)
    best_state, best_epoch = model.msd.state_dict().copy(), 0
    
    # Creates func to save sample of the models predictions and losses
    loss_tracker = LossTracker(training_error=best_val, validation_error=best_val)
    save_val = save_val_sample_pred(val_dl)
    save_val(model, 'outputs/val_ims_e0.png')

    for e in range(epochs):
        batches = tqdm(train_dl, desc=f"epoch {e+1}", position=0)
        running_loss = []

        for i, (input_, target) in enumerate(batches):
            output = model.net(input_)
            loss = loss_criterion(output, target)
            running_loss.append(loss.item())

            if regularization is not None:
                reg = regularization(output)
                (loss + reg).backward()
            else:
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            batches.set_description(f"epoch {e+1}, loss: {sum(running_loss[-2:])/2:.3e}")
            batches.update()

            if not i % 10:
                with evaluate(model):
                    # Evaluation on subset of val set upper bounded by n_val_samples
                    val_loss = sum(map(compute_loss, val_dl)) /len(val_dl)
                    tr_loss = sum(running_loss) /len(running_loss)
                    running_loss.clear()
                    loss_tracker.update(e*len(train_dl)+i+1, tr_loss, val_loss)

        if e+1 >= 10: scheduler.step()
        loss_tracker.plot(filename=f'{save_folder}/training_losses.png',
                          xticks=[i*len(train_dl) for i in range(e+2)])

        val_loss = sum(map(compute_loss, val_dl)) /len(val_dl)
        if val_loss < best_val:
            best_val, best_epoch = val_loss, e+1
            best_state = model.msd.state_dict()
        print(f"Validation loss: {val_loss:.4e}", flush=True)

        save_val(model, f'outputs/val_ims_e{e+1}.png')
        torch.save(model.msd.state_dict(), f"{save_folder}/model_epoch{e+1}.h5")

    torch.save(best_state, f"{save_folder}/best_model_{best_epoch}.h5")


def cross_validation_train():
    """Trains model with cross validation to validate stability/robustness"""
    pass
