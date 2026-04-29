import torch.nn.functional as F
from tqdm import tqdm

import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassRecall,
    MulticlassPrecision,
)

def calculate_metrics(preds, targets, num_bootstraps=1000, seed=42, num_classes=6):
    preds = np.array(preds)
    targets = np.array(targets)
    n = len(preds)

    accs, kappas, f1s, recalls, precisions, mean_error = [], [], [], [], [], []
    confusion_matrices = []

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    for _ in range(num_bootstraps):
        idxs = rng.choice(n, n, replace=True)
        p_sample = torch.tensor(preds[idxs])
        t_sample = torch.tensor(targets[idxs])

        acc = (p_sample == t_sample).float().mean().item()
        accs.append(acc)

        kappa = cohen_kappa_score(p_sample.numpy(), t_sample.numpy(), weights='quadratic')
        kappas.append(kappa)

        f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        recall = MulticlassRecall(num_classes=num_classes, average='macro')
        precision = MulticlassPrecision(num_classes=num_classes, average='macro')

        f1s.append(f1(p_sample, t_sample).item())
        recalls.append(recall(p_sample, t_sample).item())
        precisions.append(precision(p_sample, t_sample).item())
        diff = torch.abs(t_sample - p_sample).float()
        mean_error.append(diff.mean().item())
        # mean_error.append(torch.mean(torch.abs(t_sample - p_sample)).item())

        cm = confusion_matrix(t_sample.numpy(), p_sample.numpy(), labels=np.arange(num_classes))
        cm_norm = cm / cm.sum(axis=1, keepdims=True, where=(cm.sum(axis=1, keepdims=True) != 0))
        confusion_matrices.append(cm_norm)

    def summarize_metric(values):
        mean = np.mean(values)
        std = np.std(values)
        lower = np.percentile(values, 5)
        upper = np.percentile(values, 95)
        return {'mean': mean, 'std': std, 'ci_5': lower, 'ci_95': upper}

    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)

    return {
        'accuracy': summarize_metric(accs),
        'kappa': summarize_metric(kappas),
        'f1_macro': summarize_metric(f1s),
        'recall_macro': summarize_metric(recalls),
        'precision_macro': summarize_metric(precisions),
        "mean_error": summarize_metric(mean_error),
        'confusion_matrix': avg_confusion_matrix
    }

def model_checkpoint(model, best_metric, acctualy_metric, path):
    print(f"Salvando o melhor modelo... {best_metric} -> {acctualy_metric}")
    torch.save(model.state_dict(), path)


def calculate_entropy(predictions):
    """
    Calcula a entropia das predições para medir a incerteza do modelo.

    Args:
        predictions: Tensor com probabilidades (após sigmoid) de shape (batch_size, num_classes)

    Returns:
        Tensor com a entropia média por amostra de shape (batch_size,)
    """
    # Evitar log(0) adicionando um valor pequeno (epsilon)
    epsilon = 1e-10
    predictions = torch.clamp(predictions, epsilon, 1 - epsilon)

    # Entropia: -[p*log(p) + (1-p)*log(1-p)]
    entropy = -(predictions * torch.log(predictions) +
                (1 - predictions) * torch.log(1 - predictions))

    # Retornar entropia média por amostra (média sobre as classes)
    return entropy.mean(dim=1)


def compute_image_entropies(model, dataloader):
    model.eval()
    entropies = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            entropy = calculate_entropy(probs)
            entropies.extend(entropy.cpu().numpy())

    return np.array(entropies)


def remove_high_entropy_images(images, entropies, threshold=0.9):
    high_entropy_indices = np.where(entropies >= np.percentile(entropies, 100 - threshold * 100))[0]
    return [img for i, img in enumerate(images) if i not in high_entropy_indices]



def evaluation(model, dataloader, device):
    model.eval()
    model.to(device)

    bar_progress = tqdm(dataloader)

    all_logits = []
    predicts = []
    targets = []
    imgs = []

    with torch.no_grad():
        for index, (batch_data, batch_targets, img_ids) in enumerate(bar_progress):
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
            imgs.extend(img_ids)

            outputs = model(batch_data)

            prediction = outputs.sigmoid()
            prediction = prediction.sum(1).detach().round()

            all_logits.append(outputs)
            predicts.append(prediction)
            targets.append(batch_targets.sum(1))

    predicts = torch.cat(predicts).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()

    results = calculate_metrics(predicts, targets)

    return {
        "val_acc": results.get("accuracy"),
        "val_kappa": results.get("kappa"),
        "val_f1": results.get("f1_macro"),
        "val_recall": results.get("recall_macro"),
        "val_precision": results.get("precision_macro"),
        "mean_error": results.get("mean_error"),
        "confusion_matrix": results.get("confusion_matrix")
    }, (predicts, targets, imgs)


def format_metrics(metrics: dict) -> str:
    lines = []
    for metric, values in metrics.items():
        if isinstance(values, dict):
            mean = float(values["mean"])
            std = float(values["std"])
            ci_5 = float(values["ci_5"])
            ci_95 = float(values["ci_95"])

            lines.append(
                f"{metric.upper():<12} "
                f"Mean: {mean:.3f} | "
                f"Std: {std:.3f} | "
                f"95% CI: [{ci_5:.3f}, {ci_95:.3f}]"
            )
        elif metric != "confusion_matrix":
            lines.append(f"{metric.upper():<12} {values:.3f}")

    return "\n".join(lines)
