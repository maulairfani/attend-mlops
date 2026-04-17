"""ZenML step: pure metric computation over (similarities, labels).

Reports:
- tar_at_far_0.1 — True Accept Rate when False Accept Rate = 0.1%
- tar_at_far_1.0 — True Accept Rate when False Accept Rate = 1.0%
- auc           — Area Under ROC curve
"""

import numpy as np
from sklearn.metrics import auc, roc_curve
from zenml import step


def _tar_at_far(sims: list[float], labels: list[int], target_far: float) -> float:
    fpr, tpr, _ = roc_curve(labels, sims)
    idx = int(np.searchsorted(fpr, target_far))
    if idx >= len(tpr):
        return float(tpr[-1])
    return float(tpr[idx])


@step
def compute_metrics(
    similarities: list[float],
    labels: list[int],
    n_skipped: int,
    model_name: str,
) -> dict:
    if not similarities:
        raise ValueError("No similarities to compute metrics on (all pairs skipped?)")

    fpr, tpr, _ = roc_curve(labels, similarities)
    roc_auc = float(auc(fpr, tpr))

    metrics = {
        "model": model_name,
        "tar_at_far_0.1": _tar_at_far(similarities, labels, target_far=0.001),
        "tar_at_far_1.0": _tar_at_far(similarities, labels, target_far=0.01),
        "auc": roc_auc,
        "n_pairs": len(similarities),
        "n_skipped": n_skipped,
    }

    print(f"[compute_metrics:{model_name}]")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics
