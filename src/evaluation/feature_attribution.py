"""
Feature attribution utilities for PyTorch financial models.

Captum is optional. Core training and prediction code does not import Captum;
these helpers raise a clear error only when attribution is requested without the
optional dependency installed.
"""

from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import torch


SUPPORTED_ATTRIBUTION_METHODS = {"integrated_gradients", "feature_ablation"}


class AttributionDependencyError(ImportError):
    """Raised when an optional attribution dependency is missing."""


def _require_captum(method: str):
    if method == "integrated_gradients":
        try:
            from captum.attr import IntegratedGradients
        except ImportError as exc:
            raise AttributionDependencyError(
                "Captum is required for integrated_gradients attribution. "
                "Install it with `pip install captum` to use this utility."
            ) from exc
        return IntegratedGradients

    if method == "feature_ablation":
        try:
            from captum.attr import FeatureAblation
        except ImportError as exc:
            raise AttributionDependencyError(
                "Captum is required for feature_ablation attribution. "
                "Install it with `pip install captum` to use this utility."
            ) from exc
        return FeatureAblation

    raise ValueError(f"Unsupported attribution method: {method}")


def _to_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: Union[str, torch.device]) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)
        for key, value in batch.items()
    }


def _make_baseline(features: torch.Tensor, baseline: Optional[Union[str, float, torch.Tensor]]) -> torch.Tensor:
    if baseline is None or baseline == "zero":
        return torch.zeros_like(features)
    if baseline == "mean":
        return features.mean(dim=0, keepdim=True).expand_as(features)
    if isinstance(baseline, (int, float)):
        return torch.full_like(features, float(baseline))
    if isinstance(baseline, torch.Tensor):
        return baseline.to(device=features.device, dtype=features.dtype)
    raise ValueError("baseline must be None, 'zero', 'mean', a number, or a tensor")


def _feature_forward(model, fixed_batch: Dict[str, torch.Tensor]):
    def forward(features: torch.Tensor) -> torch.Tensor:
        return model(
            features=features,
            stock_id=fixed_batch["stock_id"],
            group_id=fixed_batch["group_id"],
            day=fixed_batch["day"],
            month=fixed_batch["month"],
            dividend_flag=fixed_batch["dividend_flag"],
        )

    return forward


def _attribute_features_with_captum(
    model,
    batch: Dict[str, torch.Tensor],
    method: str,
    baseline: Optional[Union[str, float, torch.Tensor]],
    target: Optional[int],
    n_steps: int,
    perturbations_per_eval: int,
) -> torch.Tensor:
    AttributionClass = _require_captum(method)
    features = batch["features"].detach().clone().requires_grad_(method == "integrated_gradients")
    baselines = _make_baseline(features, baseline)
    forward_func = _feature_forward(model, batch)
    attribution = AttributionClass(forward_func)

    if method == "integrated_gradients":
        return attribution.attribute(
            features,
            baselines=baselines,
            target=target,
            n_steps=n_steps,
        )

    return attribution.attribute(
        features,
        baselines=baselines,
        target=target,
        perturbations_per_eval=perturbations_per_eval,
    )


def attribute_batch(
    model,
    batch: Dict[str, torch.Tensor],
    method: str = "integrated_gradients",
    baseline: Optional[Union[str, float, torch.Tensor]] = "zero",
    target: Optional[int] = 0,
    n_steps: int = 16,
    perturbations_per_eval: int = 1,
    device: Optional[Union[str, torch.device]] = None,
) -> np.ndarray:
    """
    Compute raw feature attributions for one batch.

    Returns:
        NumPy array with shape (batch, seq_len, num_features).
    """
    if method not in SUPPORTED_ATTRIBUTION_METHODS:
        raise ValueError(f"method must be one of {sorted(SUPPORTED_ATTRIBUTION_METHODS)}")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if perturbations_per_eval <= 0:
        raise ValueError("perturbations_per_eval must be positive")

    if device is None:
        device = next(model.parameters()).device

    tensor_batch = _move_batch_to_device(batch, device)
    model.eval()
    attributions = _attribute_features_with_captum(
        model=model,
        batch=tensor_batch,
        method=method,
        baseline=baseline,
        target=target,
        n_steps=n_steps,
        perturbations_per_eval=perturbations_per_eval,
    )
    return _to_numpy(attributions)


def aggregate_feature_attributions(
    attributions,
    feature_names: Optional[List[str]] = None,
    use_absolute: bool = True,
) -> pd.DataFrame:
    """
    Aggregate sequence attributions into feature-level importance.

    Args:
        attributions: Array/tensor shaped (batch, seq_len, num_features).
        feature_names: Optional names for the feature dimension.
        use_absolute: Aggregate absolute attribution magnitudes if True.
    """
    values = _to_numpy(attributions)
    if values.ndim != 3:
        raise ValueError("attributions must have shape (batch, seq_len, num_features)")

    num_features = values.shape[2]
    if feature_names is None:
        feature_names = [f"feature_{idx}" for idx in range(num_features)]
    if len(feature_names) != num_features:
        raise ValueError("feature_names length must match attribution feature dimension")

    scoring_values = np.abs(values) if use_absolute else values
    importance = scoring_values.mean(axis=(0, 1))

    report = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
        "mean_attribution": values.mean(axis=(0, 1)),
        "mean_abs_attribution": np.abs(values).mean(axis=(0, 1)),
    })
    return report.sort_values("importance", ascending=False).reset_index(drop=True)


def _iter_batches(data) -> Iterable[Dict[str, torch.Tensor]]:
    if isinstance(data, dict):
        yield data
        return
    yield from data


def compute_feature_attribution_report(
    model,
    data,
    feature_names: Optional[List[str]] = None,
    method: str = "integrated_gradients",
    baseline: Optional[Union[str, float, torch.Tensor]] = "zero",
    target: Optional[int] = 0,
    n_steps: int = 16,
    perturbations_per_eval: int = 1,
    max_batches: int = 1,
    max_samples: Optional[int] = 128,
    use_absolute: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, object]:
    """
    Compute sampled feature attribution report.

    Args:
        model: PyTorch model.
        data: A batch dict or iterable/data loader yielding batch dicts.
        feature_names: Optional feature names.
        method: Captum method name.
        baseline: Baseline specification.
        target: Output target index. Regression models with shape (batch, 1)
            should generally use target=0.
        max_batches: Maximum number of batches to attribute.
        max_samples: Maximum total samples to keep in the report.
        use_absolute: Sort feature importance by absolute attribution magnitude.

    Returns:
        Dict with raw attributions, feature importance report, and metadata.
    """
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")

    attribution_chunks = []
    samples_seen = 0

    for batch_idx, batch in enumerate(_iter_batches(data)):
        if batch_idx >= max_batches:
            break
        batch_attributions = attribute_batch(
            model=model,
            batch=batch,
            method=method,
            baseline=baseline,
            target=target,
            n_steps=n_steps,
            perturbations_per_eval=perturbations_per_eval,
            device=device,
        )

        if max_samples is not None:
            remaining = max_samples - samples_seen
            if remaining <= 0:
                break
            batch_attributions = batch_attributions[:remaining]

        attribution_chunks.append(batch_attributions)
        samples_seen += batch_attributions.shape[0]

        if max_samples is not None and samples_seen >= max_samples:
            break

    if not attribution_chunks:
        raise ValueError("No batches were available for attribution")

    raw_attributions = np.concatenate(attribution_chunks, axis=0)
    feature_importance = aggregate_feature_attributions(
        raw_attributions,
        feature_names=feature_names,
        use_absolute=use_absolute,
    )

    return {
        "raw_attributions": raw_attributions,
        "feature_importance": feature_importance,
        "metadata": {
            "method": method,
            "baseline": "zero" if baseline is None else str(baseline),
            "sample_count": int(raw_attributions.shape[0]),
            "sequence_length": int(raw_attributions.shape[1]),
            "num_features": int(raw_attributions.shape[2]),
            "use_absolute": use_absolute,
            "n_steps": n_steps if method == "integrated_gradients" else None,
            "perturbations_per_eval": (
                perturbations_per_eval if method == "feature_ablation" else None
            ),
            "interpretation_note": (
                "Attributions are diagnostic, not causal proof. Compare across "
                "validation windows/regimes before using them for decisions."
            ),
        },
    }
