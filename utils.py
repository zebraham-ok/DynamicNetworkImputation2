"""
Shared utility functions: deep merge, dynamic import, auto-argument resolution, device resolution,
threshold search on a saved pair of positive/negative prediction scores.
"""
import importlib
import torch
import yaml
import os


def deep_merge(base, override):
    """Deep-merge two dicts; override takes precedence over base for keys with the same name"""
    result = {}
    all_keys = set(base.keys()) | set(override.keys())
    for key in all_keys:
        if key in override and key in base and isinstance(base[key], dict) and isinstance(override[key], dict):
            result[key] = deep_merge(base[key], override[key])
        elif key in override:
            result[key] = override[key]
        else:
            result[key] = base[key]
    return result


def import_attr(module_path, attr_name):
    """Dynamically import module.attr"""
    mod = importlib.import_module(module_path)
    return getattr(mod, attr_name)


def resolve_auto_kwargs(kwargs, context):
    """
    Resolve the auto values in kwargs:
    - "auto" string -> look up the same-named key in context
    - ["auto", 128, 64] -> the first list item is replaced by the corresponding value
    - "torch.nn.ReLU()" etc. -> evaluate
    """
    resolved = {}
    for k, v in kwargs.items():
        if v == "auto":
            resolved[k] = context.get(k, None)
        elif isinstance(v, list) and len(v) > 0 and v[0] == "auto":
            resolved[k] = [context.get(k, None)] + v[1:]
        elif isinstance(v, str) and v.startswith("torch."):
            resolved[k] = eval(v)
        else:
            resolved[k] = v
    return resolved


def resolve_device(device_str):
    """Resolve the device string; auto means automatic selection"""
    if device_str in (None, 'auto'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_str


# --- Threshold search on a saved set of prediction scores -------------------------------------

# F1 maximisation grid: identical to Prediction/find_threshold.py and
# Prediction/find_threshold_per_model.py, so the values stay comparable with their
# threshold_results.json. NOTE the half-open interval - a threshold sitting exactly at the upper
# end (e.g. the 0.899 quoted in the paper) is *not* reachable by this scan.
F1_SCAN = (0.1, 0.9, 0.001)


def youden_f1max_thresholds(pos_scores, neg_scores, f1_scan=F1_SCAN):
    """
    Report the thresholds implied by one positive/negative score pair, using exactly the convention
    already implemented by Prediction/find_threshold*.py:

    - **Youden's J**: argmax(TPR - FPR) on the ROC curve, the threshold is read off the roc_curve
      threshold grid (not re-scanned);
    - **F1-max**: grid scan over np.arange(*f1_scan) with `f1_score(..., zero_division=0)`; on ties
      the first (smallest) maximum wins, as in the reference scripts.

    Pure function: reads no config, touches no file, has no side effect. It is meant to be called
    right after a set of prediction scores has been produced (e.g. the test scores about to be
    written into a .npy) so that the operating point of that file is documented. **None of these
    values may drive model selection or early stopping** - those use the selection split only
    (see DynamicGraphTrainer.train).

    Degenerate inputs are reported rather than raising:
      - empty class / single class, or non-finite scores only -> None;
      - a constant score (nothing is separable, argmax J falls on the +inf threshold of the ROC
        curve) -> `youden_j = 1.0` with `youden_j_at_inf = True`, so the record stays strictly
        JSON-serialisable instead of holding an Infinity literal;
      - no scanned threshold reaches F1 > 0 -> the scan start is returned with `f1_at_f1max = 0.0`.

    Returns a dict of plain Python scalars, or None when the metrics are undefined.
    """
    import numpy as np
    from sklearn.metrics import roc_curve, roc_auc_score, f1_score

    pos = np.asarray(pos_scores, dtype=float).ravel()
    neg = np.asarray(neg_scores, dtype=float).ravel()
    if pos.size == 0 or neg.size == 0:
        return None

    y_true = np.concatenate([np.ones(pos.size, dtype=int), np.zeros(neg.size, dtype=int)])
    y_score = np.concatenate([pos, neg])
    finite = np.isfinite(y_score)          # nan / inf would break roc_curve
    if not finite.all():
        y_true, y_score = y_true[finite], y_score[finite]
    if y_true.size == 0 or y_true.min() == y_true.max():
        return None

    out = {
        'n_pos': int((y_true == 1).sum()),
        'n_neg': int((y_true == 0).sum()),
        'auc': float(roc_auc_score(y_true, y_score)),
    }

    fpr, tpr, roc_thr = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    best_j = int(np.argmax(j_scores))
    t_youden = float(roc_thr[best_j])
    # thresholds[0] is +inf (the "predict everything positive" point). Keep the report strictly
    # JSON-serialisable and log a flag instead of writing an Infinity literal into the record.
    out['youden_j_at_inf'] = not np.isfinite(t_youden)
    out['youden_j'] = 1.0 if out['youden_j_at_inf'] else t_youden
    out['youden_j_stat'] = float(j_scores[best_j])
    out['tpr_at_youden'] = float(tpr[best_j])
    out['fpr_at_youden'] = float(fpr[best_j])
    out['f1_at_youden'] = float(
        f1_score(y_true, (y_score >= out['youden_j']).astype(int), zero_division=0))

    start, stop, step = f1_scan
    best_f1, best_t = 0.0, None
    for t in np.arange(start, stop, step):
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    out['f1_max'] = float(start) if best_t is None else best_t
    out['f1_at_f1max'] = best_f1
    out['f1_scan'] = [float(start), float(stop), float(step)]
    return out
