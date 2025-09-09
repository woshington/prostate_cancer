from sklearn.metrics import cohen_kappa_score
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from torchmetrics import BootStrapper, CohenKappa
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassRecall,
    MulticlassPrecision,
    CohenKappa
)

# def calculate_metrics(preds, targets):
#     accuracy = (preds == targets).mean() * 100.
#     kappa = cohen_kappa_score(preds, targets, weights='quadratic')
#     quantis = torch.tensor([0.05, 0.95])
#     bootstrap = BootStrapper(
#         CohenKappa(task='multiclass', num_classes=6, weights='quadratic'),
#         num_bootstraps=1000,
#         quantile=quantis,
#         # sampling_strategy='multinomial'
#     )
#     bootstrap.update(torch.tensor(preds), torch.tensor(targets))
#     output = bootstrap.compute()
#     return accuracy, kappa, output

import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassRecall,
    MulticlassPrecision,
    CohenKappa
)

def calculate_metrics(preds, targets, num_bootstraps=1000, seed=42):
    preds = np.array(preds)
    targets = np.array(targets)
    n = len(preds)

    accs, kappas, f1s, recalls, precisions = [], [], [], [], []

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    for _ in range(num_bootstraps):
        idxs = rng.choice(n, n, replace=True)
        p_sample = torch.tensor(preds[idxs])
        t_sample = torch.tensor(targets[idxs])

        acc = (p_sample == t_sample).float().mean().item() * 100.
        accs.append(acc)

        kappa = cohen_kappa_score(p_sample.numpy(), t_sample.numpy(), weights='quadratic')
        kappas.append(kappa)

        f1 = MulticlassF1Score(num_classes=6, average='macro')
        recall = MulticlassRecall(num_classes=6, average='macro')
        precision = MulticlassPrecision(num_classes=6, average='macro')

        f1s.append(f1(p_sample, t_sample).item())
        recalls.append(recall(p_sample, t_sample).item())
        precisions.append(precision(p_sample, t_sample).item())

    def summarize_metric(values):
        # print(values)
        mean = np.mean(values)
        std = np.std(values)
        lower = np.percentile(values, 5)
        upper = np.percentile(values, 95)
        return {'mean': mean, 'std': std, f'ci_5': lower, f'ci_95': upper}

    return {
        'accuracy': summarize_metric(accs),
        'kappa': summarize_metric(kappas),
        'f1_macro': summarize_metric(f1s),
        'recall_macro': summarize_metric(recalls),
        'precision_macro': summarize_metric(precisions),
    }

def model_checkpoint(model, best_metric, acctualy_metric, path):
    print(f"Salvando o melhor modelo... {best_metric} -> {acctualy_metric}")
    torch.save(model.state_dict(), path)


def calculate_entropy(predictions=None, logits=None):
    if predictions:
        return Categorical(probs=predictions).entropy()
    return Categorical(logits=logits).entropy()


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
        "val_precision": results.get("precision_macro")
    }, (predicts, targets, imgs)


def format_metrics(metrics: dict) -> str:
    lines = []
    for metric, values in metrics.items():
        mean = float(values["mean"])
        std = float(values["std"])
        ci_5 = float(values["ci_5"])
        ci_95 = float(values["ci_95"])

        lines.append(
            f"{metric.upper():<12} "
            f"Mean: {mean:.2f} | "
            f"Std: {std:.2f} | "
            f"95% CI: [{ci_5:.2f}, {ci_95:.2f}]"
        )
    return "\n".join(lines)
