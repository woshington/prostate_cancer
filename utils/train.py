from tqdm import tqdm
import torch
import numpy as np
from utils.metrics import calculate_metrics, calculate_entropy, format_metrics
from utils.layer import MixUp, CutMix
import random


def training_step(model, dataset_loader, optimizer, epoch, device, loss_function, use_mixup=True):
    model= model.to(device)
    model.train()
    train_loss = []

    mixup = MixUp(alpha=0.2)
    cutmix = CutMix(alpha=1.0)

    bar_progress = tqdm(dataset_loader)

    for index, (batch_data, batch_targets, _) in enumerate(bar_progress):
        batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)

        optimizer.zero_grad()

        # Apply MixUp or CutMix randomly
        # if use_mixup and random.random() > 0.5:
        #     if random.random() > 0.5:
        #         batch_data, targets_a, targets_b, lam = mixup(batch_data, batch_targets)
        #     else:
        #         batch_data, targets_a, targets_b, lam = cutmix(batch_data, batch_targets)
        #
        #     logits = model(batch_data)
        #     loss = lam * loss_function(logits, targets_a) + (1 - lam) * loss_function(logits, targets_b)
        # else:
        logits = model(batch_data)
        loss = loss_function(logits, batch_targets)

        loss.backward()

        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)

        bar_progress.set_description(
            'loss: %.5f, smooth loss: %.5f' % (loss_np, smooth_loss)
        )

    return train_loss


def validation_step(model, dataset_loader, device, loss_function):
    model.eval()

    validation_loss = []
    preds = []
    targets = []

    bar_progress = tqdm(dataset_loader)  # Barra de progresso

    with torch.no_grad():
        for index, (batch_data, batch_targets, img_ids) in enumerate(bar_progress):
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)

            logits = model(batch_data)

            loss = loss_function(logits, batch_targets)

            prediction = logits.sigmoid()
            prediction = prediction.sum(1).detach().round()

            preds.append(prediction)
            targets.append(batch_targets.sum(1))

            validation_loss.append(loss.detach().cpu().numpy())

        validation_loss = np.mean(validation_loss)

    preds = torch.cat(preds).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()

    results = calculate_metrics(preds, targets)

    return {
        "val_loss": validation_loss,
        "val_acc": results.get("accuracy"),
        "val_kappa": results.get("kappa"),
        "val_f1": results.get("f1_macro"),
        "val_recall": results.get("recall_macro"),
        "val_precision": results.get("precision_macro")
    }

def train_model(
    model,
    epochs,
    optimizer,
    scheduler,
    train_dataloader,
    valid_dataloader,
    checkpoint,
    device,
    loss_function,
    path_to_save_metrics,
    path_to_save_model=None,
    patience=20,
):
    best_metric_criteria = 0.
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}\n')

        train_loss = training_step(model, train_dataloader, optimizer, epoch, device, loss_function)
        metrics = validation_step(model, valid_dataloader, device, loss_function)
        result = format_metrics(metrics)
        print(result)


        log_epoch = f'epoch: {epoch} | lr: {optimizer.param_groups[0]["lr"]:.7f} | Train loss: {np.mean(train_loss)} | Validation loss: {metrics["val_loss"]} | Validation accuracy: {metrics["val_acc"]["mean"]} | QWKappa: {metrics["val_kappa"]["mean"]}'
        with open(path_to_save_metrics, 'a') as f:
            f.write(log_epoch + '\n')

        if metrics["val_kappa"]['mean'] >= best_metric_criteria:
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint(model, best_metric_criteria, metrics["val_kappa"]['mean'], path_to_save_model)
            best_metric_criteria = metrics["val_kappa"]['mean']
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'\nEarly stopping at epoch {epoch}. No improvement for {patience} epochs.')
            print(f'Best epoch: {best_epoch} with kappa: {best_metric_criteria:.4f}')
            break
        # scheduler.step(metrics["val_loss"])
        scheduler.step()

def apply_active_learning(model, epochs, optimizer, scheduler, train_dataloader, device, loss_function):
    model.to(device)
    for epoch in range(1, epochs + 1):
        training_step(model, train_dataloader, optimizer, epoch, device, loss_function)
        scheduler.step()
        log_epoch = f'epoch: {epoch} | lr: {optimizer.param_groups[0]["lr"]:.7f}'
        # print(log_epoch)

    # return model

def remove_images_by_entropy(model, dataset_loader, device):
    model.eval()

    bar_progress = tqdm(dataset_loader)
    remove_images = {}
    with torch.no_grad():
        for index, (batch_data, batch_targets, img_ids) in enumerate(bar_progress):
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)

            logits = model(batch_data)

            batch_entropy = calculate_entropy(logits=logits)
            entropies = np.array(batch_entropy.cpu().numpy())

            remove_images.update({img_ids[index]: entropy for index, entropy in enumerate(entropies)})

    return remove_images